"""Advisory audit of a research report against its claim database (v3x, 2026-10-18). Deterministic: no LLM, no network.

usage: check_report.py <research_dir> [--strict]      -> <research_dir>/CHECK.md + one summary line

  tags      every confidence tag on a bullet / paragraph / table row: [Yüksek] (Orta) **[Düşük]** [Yüksek/Orta],
            bold **Yüksek** / **Orta:**. URLs of a unit = inline URLs + [n] / [n,m] citations resolved in the next
            numbered "## Kaynaklar" list. A Yüksek tag is supported if one URL cited in the same
            unit is in a Yüksek cluster of claims/claims.csv (own or corroborating URL; a cluster whose domains are
            all but one content-farm / unreliable does not count: claims.csv written before srcclass), or if the
            unit cites 2+ independent non-farm domains that each occur in claims.csv; otherwise "overclaimed". A tag
            with no web URL in its unit is "unsourced" (Orta / Düşük too). URLs: arXiv id / DOI, else lower-case
            host without www., no fragment / utm_, no trailing slash.
  headings  top-level "# N." / "# NN —" numbers in REPORT.md (else report/NN-*.md in file order): duplicates
            (v2 had two "# 6.") and gaps; duplicate NN- file prefixes in report/ are listed as a note only
            (assemble_report.py renumbers the headings, so they no longer cause duplicates).
  figures   fig/*.png (and report/fig/) present / referenced by name in the report; flagged when data/*.csv exist
            but there is no figure, or the report embeds a .png that does not exist.
Report and claims text is untrusted data: only parsed with regexes, never executed; every file read must resolve
inside <research_dir> (symlinks out are skipped) and is size-capped; everything echoed into CHECK.md is sanitised
(corpus.inline) and markdown-escaped. Exit 0 always; --strict exits 1 on any overclaimed / unsourced tag, heading
problem or figure flag.
"""
import argparse
import collections
import csv
import io
import json
import pathlib
import re
import sys
import urllib.parse

from .corpus import atomic_write, domain, inline, under, work_id  # noqa: E402
from .srcclass import src_class  # noqa: E402

MAX_MD, MAX_CSV, MAX_ROWS, MAX_FILES, MAX_OUT = 4 << 20, 64 << 20, 200_000, 200, 400
LEVELS = ("Yüksek", "Orta", "Düşük")
_LV = "(?:Yüksek|Orta|Düşük)"
# [Yüksek] (Orta) [Yüksek / alt grup: Düşük] [güven: Orta] **[Yüksek]**: a level word first, short tail
_TAG = re.compile(rf"[\[(]\s*(?:\*\*)?\s*(?:(?:güven|confidence)\s*:\s*)?({_LV}(?:[^\[\]()\n]{{0,60}}?))\s*(?:\*\*)?\s*[\])]")
_BOLD = re.compile(rf"\*\*({_LV})\s*:?\s*\*\*")  # "- **Yüksek** — ...", "- **Orta:** ...", "... **Yüksek** — [4,8]"
_LVW = re.compile(_LV)
# English reports (v3.1 --out-lang en): High / Medium / Low = Yüksek / Orta / Düşük. Stricter forms than the Turkish
# ones ("[High]", "(Medium confidence)", "[confidence: Low]", "**High**", "[High/Medium]"): "(low dose)" or
# "[High-resolution images]" in English prose must not count as a tag
EN2TR = {"High": "Yüksek", "Medium": "Orta", "Low": "Düşük"}
_EN = "(?:High|Medium|Low)"
_TAG_EN = re.compile(rf"[\[(]\s*(?:\*\*)?\s*(?:confidence\s*:\s*)?({_EN}(?:\s+confidence)?(?:\s*/\s*{_EN})?)\s*(?:\*\*)?\s*[\])]")
_BOLD_EN = re.compile(rf"\*\*({_EN})(?:\s+confidence)?\s*:?\s*\*\*")


def en_tags(u):
    """English confidence tags of a unit, mapped to the Turkish level words the rest of the audit uses."""
    out = [m.group(1) for m in _BOLD_EN.finditer(u)] + [m.group(1) for m in _TAG_EN.finditer(u)]
    return [re.sub(_EN, lambda m: EN2TR[m.group(0)], t) for t in out]
_URL = re.compile(r"https?://(?:[^\s<>()\[\]{}\"'`|\\]|\([^\s<>()]{0,100}\))+")  # wiki-style (x) parens kept
_START = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+|^\s*\|")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEAD = re.compile(r"^#[ \t]+(\d{1,3})[ \t]*(?:[.)]|[—–-])[ \t]+(\S.*)$")  # same rule as assemble_report.renumber
_PNG = re.compile(r"(?i)([\w./-]{0,200}?\.png)\b")
# a reference-list heading in any output language sections.SOURCES_HEAD writes (plus the usual English / Turkish forms)
_REFWORDS = (r"kaynak\w*|referans\w*|references?|sources?|bibliography|quellen|fuentes|fonti|fontes|bronnen|źródła|"
             r"источники|джерела|来源|出典|출처|المصادر|källor|kilder|lähteet|zdroje|πηγές|sumber|surse|források")
_REFHEAD = re.compile(rf"(?i)^#{{1,4}}\s*({_REFWORDS})\b")
_REFITEM = re.compile(r"^\s*(?:[-*+]\s*)?\[?(\d{1,3})[\].):]\s")
_CITE = re.compile(r"\[(\d{1,3}(?:\s*[,;–-]\s*\d{1,3}){0,20})\]")  # [4] [4,8] [3-5]: numbered reference lists
_IMG = re.compile(r"!\[[^\]\n]{0,300}\]\(\s*<?([^)\s>]{1,300})>?")


def read_text(root, rel, cap=MAX_MD):
    """File under root (resolved inside it, regular file) -> text, capped; None if absent / outside / unreadable."""
    try:
        p = under(root, rel)
        if not p.is_file():
            return None
        with open(p, "rb") as f:
            return f.read(cap).decode("utf-8", "replace")
    except (ValueError, OSError):
        return None


def listing(root, rel, pattern):
    """Names in root/rel matching pattern that resolve inside root (symlinks out of it dropped), capped, sorted."""
    try:
        d = under(root, rel)
    except ValueError:
        return []
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(pattern))[:MAX_FILES]:
        try:
            under(root, p.relative_to(root.resolve()))
            out.append(p)
        except ValueError:
            pass
    return out


def md(s, cap=160):
    """Untrusted text -> one safe markdown table cell (no links / html / emphasis / pipes)."""
    s = inline(s, cap)
    return re.sub(r"([\\`*_\[\]<>#!|])", r"\\\1", s)


def url_key(u):
    """Same work / same page across citation styles: arXiv id or DOI, else host (no www.) + path, minus fragment,
    tracking parameters and trailing slash."""
    u = u.strip().rstrip(".,;:!?*_'\"")
    try:
        w = work_id(u)
        if w:
            return w
        p = urllib.parse.urlsplit(u)
        q = "&".join(x for x in p.query.split("&") if x and not x.startswith(("utm_", "ref=", "fbclid")))
        return f"{(p.hostname or '').removeprefix('www.')}{p.path.rstrip('/')}" + (f"?{q}" if q else "")
    except ValueError:
        return None


def urls_in(text):
    """Web URLs in a unit: dotted host only (https://research/notes/x.md is a local path, not a source)."""
    out = []
    for m in _URL.finditer(text):
        u = m.group(0).rstrip(".,;:!?*_'\"")
        try:
            host = urllib.parse.urlsplit(u).hostname or ""
        except ValueError:
            continue
        if "." in host and url_key(u):
            out.append(u)
    return out


def load_rows(f):
    """claims.csv rows, bounded like load_claims (size cap, row cap, field cap); a malformed tail ends the list."""
    with open(f, "rb") as fh:
        t = fh.read(MAX_CSV).decode("utf-8", "replace")
    csv.field_size_limit(1 << 20)
    out = []
    try:
        for i, r in enumerate(csv.DictReader(io.StringIO(t))):
            if i >= MAX_ROWS:
                break
            out.append(r)
    except csv.Error:
        pass
    return out


def load_claims(root):
    """claims.csv -> (keys of supported Yüksek clusters, keys of farm-only Yüksek clusters, every domain cited)."""
    t = read_text(root, "claims/claims.csv", MAX_CSV)
    high, farm_high, doms = set(), set(), set()
    if t is None:
        return None
    csv.field_size_limit(1 << 20)
    try:
        for i, r in enumerate(csv.DictReader(io.StringIO(t))):
            if i >= MAX_ROWS:
                break
            us = [u for u in f"{r.get('urls') or ''} {r.get('corroborated_by') or ''}".split() if u.startswith("http")]
            ds = {d.strip().lower() for d in (r.get("domains") or "").split(",") if d.strip()} | {domain(u) for u in us}
            doms |= ds
            if r.get("confidence") == "Yüksek":
                try:
                    n = int(r.get("sources") or 0)
                except ValueError:
                    n = 0
                real = sum(not src_class(f"https://{d}/") for d in ds)
                (high if min(n, real) >= 2 else farm_high).update(k for k in map(url_key, us) if k)
    except csv.Error:
        pass  # a torn / hostile file: what was read so far is used
    return high, farm_high - high, doms


def units(text):
    """(line_no, text) per bullet (+ its continuation lines), paragraph, table row or heading; fenced code skipped."""
    out, cur, fence = [], None, False
    for i, line in enumerate(text.split("\n"), 1):
        if _FENCE.match(line):
            fence = not fence
            cur = None
            continue
        if fence:
            continue
        if not line.strip():
            cur = None
        elif line.lstrip().startswith("#") or _START.match(line) or cur is None:
            cur = [i, line]
            out.append(cur)
            if line.lstrip().startswith(("#", "|")):
                cur = None
        else:
            cur[1] += "\n" + line
    return [(n, t) for n, t in out]


def ref_blocks(us):
    """[(first line, {n: [urls]})] for each numbered "## Kaynaklar" list (an entry's continuation lines included)."""
    out, cur = [], None
    for line_no, t in us:
        if t.lstrip().startswith("#"):
            cur = {} if _REFHEAD.match(t.lstrip()) else None
            if cur is not None:
                out.append((line_no, cur))
            continue
        m = _REFITEM.match(t) if cur is not None else None
        if m:
            cur.setdefault(int(m.group(1)), []).extend(urls_in(t))
    return out


def cited_refs(u, line_no, blocks):
    """URLs behind [n] / [n,m] / [n-m] citations: the first numbered reference list after the unit."""
    refs = next((b for start, b in blocks if start > line_no), None)
    if not refs:
        return []
    ns = set()
    for m in _CITE.finditer(u):
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            a, _, b = part.replace("–", "-").partition("-")
            lo, hi = int(a), int(b or a)
            ns.update(range(lo, min(hi, lo + 50) + 1))
    return [x for n in sorted(ns) for x in refs.get(n, [])]


def audit_tags(text, claims):
    high, farm_high, doms = claims or (set(), set(), set())
    rows, us_all = [], units(text)
    blocks = ref_blocks(us_all)
    for line_no, u in us_all:
        tags = [m.group(1) for m in _BOLD.finditer(u)] + [m.group(1) for m in _TAG.finditer(u)] + en_tags(u)
        if not tags:
            continue
        us = urls_in(u) + cited_refs(u, line_no, blocks)
        keys = {url_key(x) for x in us}
        real = {domain(x) for x in us if not src_class(x)}
        farms = sorted({domain(x) for x in us if src_class(x)})
        for tag in tags:
            lv = [x for x in LEVELS if x in _LVW.findall(tag)]
            lv = lv[0] if lv else "Orta"
            if not us:
                st, why = "unsourced", "no web URL in this bullet / paragraph"
            elif lv != "Yüksek":
                st, why = "ok", ""
            elif keys & high:
                st, why = "ok", "URL in a Yüksek cluster"
            elif len(real & doms) >= 2:
                st, why = "ok", f"{len(real & doms)} independent non-farm domains in claims.csv"
            else:
                bits = ["Yüksek cluster only via content farms"] if keys & farm_high else []
                bits += [f"farm/unreliable: {', '.join(farms)}"] if farms else []
                bits += [f"{len(real & doms)} non-farm domain(s) in claims.csv"]
                st, why = "overclaimed", "; ".join(bits)
            rows.append({"line": line_no, "tag": tag.strip(), "level": lv, "status": st, "why": why,
                         "urls": len(us), "text": u})
    return rows


def audit_headings(text):
    nums, fence = [], False
    for i, line in enumerate(text.split("\n"), 1):
        if _FENCE.match(line):
            fence = not fence
        m = None if fence else _HEAD.match(line)
        if m:
            nums.append((int(m.group(1)), i, m.group(2)))
    issues = []
    cnt = collections.Counter(n for n, _, _ in nums)
    for n, c in sorted(cnt.items())[:50]:
        if c > 1:
            issues.append(f"duplicate section number {n} ({c}x): " +
                          "; ".join([f"L{i} {md(h, 60)}" for k, i, h in nums if k == n][:10]))
    if nums:
        lo = min(1, min(cnt))
        miss = [n for n in range(lo, max(cnt) + 1) if n not in cnt]
        if miss:
            issues.append("missing section number(s): " + ", ".join(map(str, miss[:50])))
    return nums, issues


def audit_figures(root, text):
    pngs = listing(root, "fig", "*.png") + listing(root, "report/fig", "*.png")
    names = {p.name for p in pngs}
    cited = {pathlib.PurePosixPath(m.group(1)).name for m in _PNG.finditer(text)}
    embeds = [m.group(1) for m in _IMG.finditer(text) if m.group(1).lower().endswith(".png")]
    missing = []
    for e in embeds:
        ok = False
        for base in ((), ("report",)):  # REPORT.md embeds fig/x.png, a section file ../fig/x.png
            try:
                ok = ok or ("://" not in e and under(root, *base, e).is_file())
            except ValueError:
                pass
        if not ok:
            missing.append(e)
    csvs = listing(root, "data", "*.csv")
    flags = []
    if csvs and not pngs:
        flags.append(f"{len(csvs)} data/*.csv but no figure (fig/*.png)")
    if missing:
        flags.append("embedded .png not found: " + ", ".join(md(x, 80) for x in missing[:20]))
    unref = sorted(names - cited)
    return {"png": len(pngs), "referenced": len(names & cited), "csv": len(csvs), "unreferenced": unref,
            "flags": flags}


def report_text(root):
    """REPORT.md if present, else report/NN-*.md joined in file order; plus duplicate NN- prefixes in report/."""
    parts = listing(root, "report", "[0-9][0-9]-*.md")
    pre = collections.Counter(p.name[:2] for p in parts)
    dup = [f"report/ files share prefix {k}-: " + ", ".join(md(p.name, 60) for p in parts if p.name[:2] == k)
           for k, c in sorted(pre.items()) if c > 1]
    t = read_text(root, "REPORT.md")
    if t is not None:
        return "REPORT.md", t, dup
    texts = [read_text(root, p.relative_to(root.resolve())) or "" for p in parts]
    return ("report/*.md" if parts else None), "\n\n".join(x.strip() for x in texts), dup


DEPTH_FLOOR = {"normal": 15000, "derin": 45000, "maksimum": 90000,  # report words per profile (v1 baseline: 8432)
               "dinamik": 15000}  # dinamik: harvest/depth.json "depth_floor" (grows with the verified yield)
SUB_MIN = 700  # words per "## " sub-topic section (Kaynaklar/Özet sections excluded)


DUP_JACCARD, DUP_SHINGLE = 0.5, 6   # two paragraphs sharing >= half their 6-word shingles = one is a repeat
THIN_WPE = 120                      # words per evidence unit (URL, [n] citation, number) above this = thin / padded
NGRAM, NGRAM_REPEAT = 8, 4          # an 8-word phrase in >= 4 paragraphs = boilerplate


def paragraphs(text, min_words=40):
    """[(section heading, paragraph text)] of prose paragraphs >= min_words (tables, code, headings, ref lists out)."""
    out, sec, buf, fence = [], "", [], False

    def flush():
        p = " ".join(buf).strip()
        if len(p.split()) >= min_words and not _REFHEAD.match("## " + sec):
            out.append((sec, p))
        buf.clear()
    for line in text.split("\n"):
        if _FENCE.match(line):
            fence = not fence; flush(); continue
        if fence or line.lstrip().startswith("|"):
            flush(); continue
        if line.startswith("#"):
            flush(); sec = line.lstrip("#").strip(); continue
        if not line.strip():
            flush(); continue
        buf.append(line.strip())
    flush()
    return out


def _words(s):
    return re.findall(r"\w+", s.lower())


def audit_padding(root, text):
    """Anti-padding / repetition / coverage audit (v3.1): near-duplicate paragraphs across the report, thin sections
    (words per evidence unit), boilerplate phrases repeated in many paragraphs, and the share of Yüksek clusters the
    report cites. Returns {"dups", "thin", "phrases", "coverage", "flags"}."""
    paras = paragraphs(text)
    sh = [set(zip(*(w[i:] for i in range(DUP_SHINGLE)))) for w in (_words(p) for _, p in paras)]
    index = collections.defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            index[g].append(i)
    shared = collections.Counter()
    for ids in index.values():
        if 1 < len(ids) <= 50:  # a shingle in > 50 paragraphs is a stock phrase, not evidence of a copy
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    shared[(ids[x], ids[y])] += 1
    dups = []
    for (i, j), n in shared.items():
        if n >= 5:
            jac = n / max(1, len(sh[i] | sh[j]))
            if jac >= DUP_JACCARD:
                dups.append((round(jac, 2), paras[i][0], paras[j][0], paras[j][1][:160]))
    dups.sort(reverse=True)
    secs = collections.defaultdict(list)
    for s, p in paras:
        secs[s].append(p)
    thin = []
    for s, ps in secs.items():
        if re.search(r"(?i)kaynak|özet|ozet|sources|summary|references", s):
            continue
        body = " ".join(ps)
        w = len(body.split())
        ev = len(urls_in(body)) + len(re.findall(r"\[\d{1,3}(?:[,\-–]\s*\d{1,3})*\]", body)) + \
            len(re.findall(r"(?<![\w.])\d+(?:[.,]\d+)?%?", body))
        if w >= 300 and w / max(1, ev) > THIN_WPE:
            thin.append((s, w, ev))
    grams = collections.defaultdict(set)
    for i, (_, p) in enumerate(paras):
        w = _words(p)
        for k in range(len(w) - NGRAM + 1):
            grams[" ".join(w[k:k + NGRAM])].add(i)
    phrases = sorted(((len(v), g) for g, v in grams.items() if len(v) >= NGRAM_REPEAT), reverse=True)[:20]
    cov = None
    f = root / "claims/claims.csv"
    if f.exists():
        try:  # a hostile / oversized claims.csv must not break the audit
            cited = {url_key(u) for u in urls_in(text)}
            hi = [r for r in load_rows(f) if r.get("confidence") == "Yüksek"]
            got = sum(1 for r in hi if any(url_key(u) in cited for u in
                                           (str(r.get("urls") or "") + " " + str(r.get("corroborated_by") or "")).split()))
            cov = (got, len(hi))
        except (csv.Error, OSError, ValueError):
            cov = None
    flags = []
    if dups:
        flags.append(f"{len(dups)} near-duplicate paragraph pairs (Jaccard >= {DUP_JACCARD}): cut the repeat, cross-reference")
    if thin:
        flags.append(f"{len(thin)} thin sections (> {THIN_WPE} words per URL / citation / number): replace filler with evidence or shorten")
    if len(phrases) >= 5:
        flags.append(f"{len(phrases)} boilerplate phrases repeated in >= {NGRAM_REPEAT} paragraphs")
    note = [f"coverage: only {cov[0]}/{cov[1]} Yüksek clusters cited ({100 * cov[0] // cov[1]}%)"] \
        if cov and cov[1] and cov[0] / cov[1] < 0.6 else []  # advisory: shown, never fails --strict
    return {"dups": dups, "thin": thin, "phrases": phrases, "coverage": cov, "flags": flags, "notes": note}


# v3.1 citation / scope discipline (advisory: listed in CHECK.md, the phase-6 fixers clear them)
UNCITED_MIN = 60  # words: a prose paragraph this long with no URL / [n] citation / ⟨S..⟩ id is uncited
# Whether evidence is outside the question's scope depends on the question, so code cannot tell (a fixed field list
# only fit one topic); the writers label analogies themselves ("by analogy from <X>", sections.RULES) and verify_fix
# asks the model per section. What code can check without knowing the field: a statement that calls itself an
# analogy must not carry a High tag.
ANALOGY = re.compile(r"(?i)\b(by analogy|analog(?:y|ies|ous|ously)|analoji\w*|benzetme\w*|alan dışı)\b")


def high_tagged(u):
    """True if a unit carries a High / Yüksek confidence tag (the tag forms audit_tags counts, not the word 'high')."""
    tags = [m.group(1) for m in _BOLD.finditer(u)] + [m.group(1) for m in _TAG.finditer(u)] + en_tags(u)
    return any(t.strip().startswith("Yüksek") for t in tags)
_NUM = re.compile(r"(?<![\w.⟨-])(\d+(?:[.,]\d+)?)\s?(%|bp|bps|x)?(?![\w⟩])")


def _num_norm(n):
    return n.replace(",", ".").rstrip("0").rstrip(".") if "." in n.replace(",", ".") else n


def audit_discipline(root, text):
    """uncited paragraphs; statements labelled as an analogy that still carry a High tag; numbers in the prose that
    appear in no claim / quote / local fact / data csv (possible invention)."""
    paras = paragraphs(text, 8)  # analogy / number checks look at short paragraphs and bullets too
    uncited = [(s, p[:140]) for s, p in paras if len(p.split()) >= UNCITED_MIN and not urls_in(p)
               and not re.search(r"\[\d{1,3}(?:[,\-–]\s*\d{1,3})*\]|⟨S\d+-\d+⟩|see \d|bkz\.? \d", p)]
    ana = [p[:160] for _, p in paras if ANALOGY.search(p) and high_tagged(p)]
    known = set()
    for rel in ("claims/claims.csv", "claims/local-facts.md", "claims/numeric-findings.md"):
        t = read_text(root, rel, MAX_CSV)
        if t:
            known |= {_num_norm(m.group(1)) for m in _NUM.finditer(t)}
    for f in sorted((root / "data").glob("*.csv"))[:MAX_FILES] if (root / "data").is_dir() else []:
        t = read_text(root, f"data/{f.name}", MAX_CSV) or ""
        known |= {_num_norm(m.group(1)) for m in _NUM.finditer(t)}
    unknown = []
    if known:
        for s, p in paras:
            for m in _NUM.finditer(p):
                n = _num_norm(m.group(1))
                if len(n.replace(".", "")) < 2 or re.fullmatch(r"(19|20)\d\d", n):  # years, single digits: too common
                    continue
                if n not in known:
                    unknown.append((s, m.group(0), p[max(0, m.start() - 60):m.end() + 40]))
    return {"uncited": uncited, "analogy_high": ana, "unknown_numbers": unknown}


def audit_depth(root, text):
    """Report depth vs the profile in harvest/profile.json: total words, short sub-topic sections, tables."""
    prof, has = "normal", (root / "harvest" / "profile.json").is_file()
    try:
        j = json.loads(read_text(root, "harvest/profile.json", 1 << 16) or "{}")
        prof = {"fast": "normal", "deep": "derin", "max": "maksimum", "dynamic": "dinamik"}.get(
            str(j.get("profile")), str(j.get("profile")))
    except (ValueError, AttributeError):
        pass
    prof = prof if prof in DEPTH_FLOOR else "normal"
    # dinamik (v3.1): no word floor and no cap - length follows the evidence (coverage + anti-padding audits below);
    # a floor made writers pad once the evidence ran out
    floor = None if prof == "dinamik" else DEPTH_FLOOR[prof]
    words = len(text.split())
    secs, cur, buf = [], None, []
    for line in text.split("\n") + ["## _end"]:
        if line.startswith("## ") or line.startswith("# "):
            if cur is not None and not re.search(rf"(?i){_REFWORDS}|özet|ozet|summary", cur):
                secs.append((cur, len(" ".join(buf).split())))
            cur, buf = (line[3:] if line.startswith("## ") else None), []
        else:
            buf.append(line)
    short = [(h, n) for h, n in secs if n < SUB_MIN]
    tables = len(re.findall(r"(?m)^\|\s*:?-{3,}", text))
    flags = [f"rapor {words} kelime < {prof} hedefi {floor}"] if floor and words < floor else []
    if floor and secs and len(short) > len(secs) // 3:  # dinamik: a short section with all its evidence is fine
        flags.append(f"{len(short)}/{len(secs)} alt başlık < {SUB_MIN} kelime")
    return {"profile": prof, "words": words, "floor": floor, "sections": len(secs),
            "short": short, "tables": tables, "flags": flags, "enforced": has}


def main(argv=None):
    ap = argparse.ArgumentParser(description="advisory report audit -> CHECK.md")
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--strict", action="store_true", help="exit 1 when anything is flagged")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if not (root / "REPORT.md").exists() and not (root / "report").is_dir() and (root / "research").is_dir():
        root = (root / "research").resolve()  # project dir given instead of research dir
    src, text, dup = report_text(root)
    if src is None:
        print(f"check_report: no REPORT.md or report/NN-*.md under {root}")
        return 1 if a.strict else 0
    claims = load_claims(root)
    tags = audit_tags(text, claims)
    nums, h_issues = audit_headings(text)
    fig = audit_figures(root, text)
    dep = audit_depth(root, text)
    pad = audit_padding(root, text)
    dis = audit_discipline(root, text)
    st = collections.Counter(r["status"] for r in tags)
    lv = collections.Counter((r["level"], r["status"]) for r in tags)
    n_y = sum(v for (lvl, _), v in lv.items() if lvl == "Yüksek")
    bad = [r for r in tags if r["status"] != "ok"]
    summ = (f"check_report: {len(tags)} tags (Yüksek {n_y}): ok {st['ok']}, overclaimed {st['overclaimed']}, "
            f"unsourced {st['unsourced']}; sections {len(nums)}, heading issues {len(h_issues)}; "
            f"figures {fig['png']} ({fig['referenced']} referenced), data csv {fig['csv']}, "
            f"figure flags {len(fig['flags'])}; words {dep['words']}/{dep['floor'] or 'evidence-driven'} ({dep['profile']}), "
            f"padding flags {len(pad['flags'])}, Yüksek coverage {pad['coverage']}, uncited {len(dis['uncited'])}, "
            f"analogy tagged High {len(dis['analogy_high'])}, unknown numbers {len(dis['unknown_numbers'])}, "
            f"tables {dep['tables']}, short sub-topics {len(dep['short'])}/{dep['sections']}" + ("" if claims is not None else "; NO claims/claims.csv"))
    out = ["# Rapor denetimi (otomatik, check_report.py)", "",
           f"Kaynak: {src}; iddia veritabanı: {'claims/claims.csv' if claims is not None else 'YOK'}. "
           "Tavsiye niteliğinde: 'overclaimed' etiketleri CLAIMS.md'nin desteklediği seviyeye indir ya da Yüksek "
           "kümedeki / 2+ bağımsız (içerik çiftliği olmayan) alan adındaki kaynağı aynı maddeye ekle; 'unsourced' "
           "etiketlere aynı maddede URL ekle.", "",
           "| ölçü | sayı |", "|---|---|",
           f"| etiket (toplam) | {len(tags)} |", f"| Yüksek etiket | {n_y} |",
           f"| Yüksek ok | {lv[('Yüksek', 'ok')]} |", f"| overclaimed (Yüksek) | {st['overclaimed']} |",
           f"| unsourced (tüm seviyeler) | {st['unsourced']} |", f"| Orta/Düşük ok | {st['ok'] - lv[('Yüksek', 'ok')]} |",
           f"| numaralı bölüm | {len(nums)} |", f"| başlık sorunu | {len(h_issues)} |",
           f"| fig/*.png | {fig['png']} |", f"| raporda geçen figür | {fig['referenced']} |",
           f"| data/*.csv | {fig['csv']} |",
           f"| kelime / hedef ({dep['profile']}) | {dep['words']} / {dep['floor'] or 'kanıta göre (taban yok)'} |",
           f"| tekrar paragraf çifti | {len(pad['dups'])} |", f"| ince bölüm | {len(pad['thin'])} |",
           f"| Yüksek küme kapsamı | {pad['coverage'][0] if pad['coverage'] else '-'} / {pad['coverage'][1] if pad['coverage'] else '-'} |",
           f"| tablo | {dep['tables']} |", f"| kısa alt başlık (<{SUB_MIN} kelime) | {len(dep['short'])}/{dep['sections']} |",
           "", "## Derinlik", ""]
    out += [f"- {x}" for x in dep["flags"]] or ["- hedef karşılandı"]
    out += [f"- kısa: {md(h, 70)} ({n} kelime)" for h, n in dep["short"][:40]]
    out += ["", "## Dolgu, tekrar ve kapsam", ""]
    out += [f"- {x}" for x in pad["flags"] + pad["notes"]] or ["- sorun yok"]
    out += [f"- tekrar ({j}): '{md(a, 50)}' ↔ '{md(b, 50)}': {md(t, 160)}" for j, a, b, t in pad["dups"][:40]]
    out += [f"- ince: '{md(s, 70)}' {w} kelime / {e} kanıt birimi" for s, w, e in pad["thin"][:40]]
    out += [f"- kalıp ifade ({n} paragraf): \"{md(g, 90)}\"" for n, g in pad["phrases"][:20]]
    out += ["", "## Atıf ve kapsam disiplini", "",
            f"- atıfsız paragraf (>= {UNCITED_MIN} kelime, URL / [n] / ⟨S..⟩ yok): {len(dis['uncited'])}",
            f"- analoji olarak yazılmış ama High etiketli ifade: {len(dis['analogy_high'])}",
            f"- kanıtta bulunmayan sayı (claims / local-facts / data/*.csv): {len(dis['unknown_numbers'])}"]
    out += [f"- atıfsız: '{md(s, 60)}': {md(p, 140)}" for s, p in dis["uncited"][:40]]
    out += [f"- analoji + High: {md(p, 160)}" for p in dis["analogy_high"][:30]]
    out += [f"- sayı {md(n, 20)} ('{md(s, 50)}'): …{md(c, 110)}…" for s, n, c in dis["unknown_numbers"][:50]]
    out += ["", "## Bölüm numaraları", ""]
    out += [f"- {x}" for x in h_issues] or ["- sorun yok: " + ", ".join(str(n) for n, _, _ in nums)]
    out += [f"- not: {x} (assemble_report.py başlıkları sırayla numaralar)" for x in dup]
    out += ["", "## Figürler", ""]
    out += [f"- {x}" for x in fig["flags"]] or ["- sorun yok"]
    if fig["unreferenced"]:
        out.append("- raporda geçmeyen: " + ", ".join(md(x, 80) for x in fig["unreferenced"][:30]))
    out += ["", f"## İşaretli etiketler ({len(bad)}{', ilk ' + str(MAX_OUT) if len(bad) > MAX_OUT else ''})", "",
            "| satır | etiket | durum | URL | neden | metin |", "|---|---|---|---|---|---|"]
    out += [f"| {r['line']} | {md(r['tag'], 40)} | {r['status']} | {r['urls']} | {md(r['why'], 200)} | {md(r['text'])} |"
            for r in bad[:MAX_OUT]]
    atomic_write(root / "CHECK.md", "\n".join(out) + "\n")
    print(summ + f" -> {root / 'CHECK.md'}")
    flagged = st["overclaimed"] or st["unsourced"] or h_issues or fig["flags"] or (dep["enforced"] and dep["flags"]) \
        or pad["flags"]
    return 1 if a.strict and flagged else 0


if __name__ == "__main__":
    sys.exit(main())
