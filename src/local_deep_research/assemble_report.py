"""Assemble research/report/NN-*.md section files into research/REPORT.md (sorted by file name).

usage: python -m local_deep_research.assemble_report <research_dir>

Top-level numbered headings ("# 6. Doğrulama", "# 06 — Modeller") are renumbered sequentially in file order, so a
late 06-dogrulama.md next to 06-karsi-gorusler.md cannot give two "# 6." sections (a live run). The first
numbered heading keeps its number (0- or 1-based) and zero-padding; separator and heading text are kept; headings
inside ``` fences and "##" subsections are left alone. Section files themselves are never rewritten.

A line (bullet / paragraph / table row) that cites evidence ids (⟨S2-0120⟩) but carries no web URL gets the source
URL of each id from the evidence pack (claims/subq-*.md) next to it, so the reader can check it (live run
2026-10-21: 71 of 189 confidence tags were "unsourced" only because writers cited ids without URLs).
"""
import pathlib
import re
import sys

_NUM_HEAD = re.compile(r"^#[ \t]+(\d{1,3})([ \t]*(?:[.)]|[—–-])[ \t]+)(\S.*)$")  # not "# 1.5 x", "# 186 ETF"
_FENCE = re.compile(r"^\s*(```|~~~)")
_CITE = re.compile(r"⟨(S\d+-\d+)⟩")
_PACK_LINE = re.compile(r"\[([^\[\]]*https?://[^\[\]]*)\]\s*⟨(S\d+-\d+)⟩\s*$")


def renumber(texts):
    """[section text] -> ([renumbered text], [(old, new, heading)]) with one running counter over all sections."""
    n, out, changes = None, [], []
    for t in texts:
        lines, fence = t.split("\n"), False
        for i, line in enumerate(lines):
            if _FENCE.match(line):
                fence = not fence
            m = None if fence else _NUM_HEAD.match(line)
            if not m:
                continue
            old, sep, rest = m.groups()
            n = int(old) if n is None else n + 1
            new = str(n).zfill(len(old)) if old.startswith("0") and len(old) > 1 else str(n)
            if new != old:
                changes.append((old, new, rest[:80]))
            lines[i] = f"# {new}{sep}{rest}"
        out.append("\n".join(lines))
    return out, changes


def pack_urls(root):
    """{evidence id: [source URL, ...]} from the evidence pack lines "... [url, url] ⟨S2-0120⟩" """
    from .srcclass import urls_in_text
    out = {}
    for f in sorted((root / "claims").glob("subq-*.md")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _PACK_LINE.search(line)
            if m:
                out.setdefault(m.group(2), urls_in_text(m.group(1)))
    return out


def cite_urls(texts, urls):
    """([text], {id: [url]}) -> ([text], n lines given URLs): a line with evidence ids and no URL gets the first URL
    of each id after it; ids without a known URL, fenced code and headings are left as they are"""
    from .srcclass import urls_in_text
    out, n = [], 0
    for t in texts:
        lines, fence = t.split("\n"), False
        for i, line in enumerate(lines):
            if _FENCE.match(line):
                fence = not fence
            if fence or line.startswith("#") or not _CITE.search(line) or urls_in_text(line):
                continue
            seen = set()

            def add(m, seen=seen):
                u = next(iter(urls.get(m.group(1)) or []), None)
                if not u or u in seen:
                    return m.group(0)
                seen.add(u)
                return f"({u}) {m.group(0)}"
            new = _CITE.sub(add, line)
            if new != line:
                lines[i], n = new, n + 1
        out.append("\n".join(lines))
    return out, n


def main(argv):
    """argv as sys.argv: [program, research_dir]"""
    root = pathlib.Path(argv[1] if len(argv) > 1 else "research").resolve()
    if root.name != "research" and (root / "research").is_dir():
        root = root / "research"
    parts = sorted((root / "report").glob("[0-9][0-9]-*.md"))
    if not parts:
        sys.exit(f"no section files in {root / 'report'}")
    texts, changes = renumber([p.read_text(encoding="utf-8").strip() for p in parts])
    texts, cited = cite_urls(texts, pack_urls(root))
    (root / "REPORT.md").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    print(f"REPORT.md <- {len(parts)} sections: {', '.join(p.name for p in parts)}")
    if cited:
        print(f"  {cited} lines cited evidence ids without a URL: source URLs added from the evidence pack")
    for old, new, head in changes:
        print(f"  renumbered: # {old} -> # {new}  {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
