"""Evidence appendix: EVERY grounded claim cluster of the run, so nothing the harvest verified is left out of the
deliverable (v3.1). Code only, no LLM: research/APPENDIX.md (+ APPENDIX.pdf via md2pdf.py when it works).

usage: appendix.py <research_dir> [--no-pdf]

Grouped by sub-question in pack order (Yüksek first), one table row per cluster: id ⟨S<n>-NNNN⟩ (the id the analysts
assign to sub-topics), confidence, sources, claim, source URL, and whether the main report cites that source.
Labels follow harvest/profile.json "out_lang" (en default, tr). Claims come from untrusted web pages: shown as data.
"""
import json
import pathlib
import re
import subprocess
import sys

from .corpus import atomic_write, inline, urls_in_text  # noqa: E402

_LINE = re.compile(r"^- \*\*(\w+)\*\* \((\d+) [^)]*\) (.+?) — \"(.*)\" \[(\S+)\] ⟨(S\d+-\d+)⟩\s*$")
TXT = {"en": {"title": "Evidence appendix", "intro": "Every verified claim cluster of the research (code-generated, "
              "not written by a model), grouped by sub-question, best first. High = 2+ independent sources incl. a "
              "scholarly / official one; Medium = one source. 'In report' = the main report cites this source. Claims "
              "are extractions from web sources: check the source before relying on one.",
              "lv": {"Yüksek": "High", "Orta": "Medium", "Düşük": "Low"}, "head": "| id | level | src | claim | source | in report |",
              "n": "{n} clusters ({h} High)", "yes": "yes",
              "other": "Low relevance to every sub-question (kept for completeness)"},
       "tr": {"title": "Kanıt eki", "intro": "Araştırmanın doğrulanmış bütün iddia kümeleri (kod üretti, model yazmadı), "
              "alt soruya göre, en iyisi önce. Yüksek = 2+ bağımsız kaynak, biri akademik / resmi; Orta = tek kaynak. "
              "'Raporda' = ana rapor bu kaynağa atıf veriyor. İddialar web kaynaklarından çıkarıldı: güvenmeden önce "
              "kaynağı kontrol et.", "lv": {}, "head": "| id | düzey | kaynak | iddia | kaynak URL | raporda |",
              "n": "{n} küme ({h} Yüksek)", "yes": "evet",
              "other": "Hiçbir alt soruya yakın olmayanlar (eksiksizlik için)"}}


def cell(s, n):
    return inline(s, n).replace("|", "/")


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else "research").resolve()
    try:
        ol = json.loads((root / "harvest/profile.json").read_text(encoding="utf-8")).get("out_lang") or "en"
    except (OSError, ValueError):
        ol = "en"
    L = TXT.get(ol, TXT["en"])
    report = (root / "REPORT.md").read_text(errors="replace", encoding="utf-8") if (root / "REPORT.md").exists() else ""
    cited = set(urls_in_text(report))
    try:
        subs = json.loads((root / "harvest/plan.json").read_text(encoding="utf-8"))["subquestions"]
    except (OSError, ValueError, KeyError):
        subs = [{"id": int(m.group(1)), "title": f"S{m.group(1)}"} for p in sorted((root / "claims").glob("subq-*.md"))
                if (m := re.fullmatch(r"subq-(\d+)\.md", p.name))]
    md = [f"# {L['title']}", "", L["intro"], ""]
    total = high = 0
    all_rows = []
    for s in subs:
        n = int(s["id"])
        files = [root / f"claims/subq-{n}.md"] + sorted((root / "claims").glob(f"subq-{n}-p*.md"),
                                                         key=lambda p: int(re.sub(r"\D", "", p.stem.split("-p")[-1]) or 0))
        rows = []
        for f in files:
            if f.exists():
                rows += [m.groups() for m in map(_LINE.match, f.read_text(errors="replace", encoding="utf-8").splitlines()) if m]
        all_rows += rows
        h = sum(r[0] == "Yüksek" for r in rows)
        total += len(rows); high += h
        md += [f"## S{n}: {cell(s.get('title', ''), 120)}", "", L["n"].format(n=len(rows), h=h), "", L["head"],
               "|---|---|---|---|---|---|"]
        md += [f"| {cid} | {L['lv'].get(lv, lv)} | {src} | {cell(claim, 400)} | {cell(url, 200)} | "
               f"{L['yes'] if url in cited else ''} |" for lv, src, claim, _q, url, cid in rows]
        md.append("")
    seen = {r[2] for r in all_rows}
    other = []
    try:  # grounded clusters the packs left out as off-topic (similarity floor): listed, so nothing is lost
        import csv
        csv.field_size_limit(1 << 24)
        for r in csv.DictReader(open(root / "claims/claims.csv", encoding="utf-8", errors="replace")):
            c = inline(r.get("claim", ""), 500)
            if r.get("confidence") in ("Yüksek", "Orta") and c not in seen:
                other.append((r["confidence"], r.get("sources", ""), c, (r.get("urls") or "").split(" ")[0]))
    except (OSError, KeyError):
        pass
    if other:
        md += [f"## {L['other']}", "", L["n"].format(n=len(other), h=sum(o[0] == "Yüksek" for o in other)), "",
               L["head"], "|---|---|---|---|---|---|"]
        md += [f"| X-{k:04d} | {L['lv'].get(lv, lv)} | {src} | {cell(c, 400)} | {cell(u, 200)} | "
               f"{L['yes'] if u in cited else ''} |" for k, (lv, src, c, u) in enumerate(other, 1)]
        total += len(other)
    out = root / "APPENDIX.md"
    atomic_write(out, "\n".join(md) + "\n")
    print(f"appendix: {total} clusters ({high} High) in {len(subs)} sub-questions -> {out}")
    if "--no-pdf" not in argv:
        from .corpus import module_cmd  # -m, not the file path: md2pdf uses package-relative imports
        cmd, env = module_cmd("md2pdf", str(out))
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
        print(f"  pdf: {'ok' if r.returncode == 0 else 'failed: ' + r.stderr[-200:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
