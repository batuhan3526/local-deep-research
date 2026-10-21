"""Extra report formats from a research folder.
Usage: python -m local_deep_research.export_extras research/
Writes REPORT.xlsx (one sheet per data/*.csv + Sources) and REPORT.pptx (title, summary, one slide per chart, sources).
Both are optional extras (pip install 'local-deep-research[reports]': pandas + openpyxl, python-pptx): a format whose
libraries are missing is skipped with a note, the other is still written, and the exit code stays 0."""
import re, sys, pathlib

from .srcclass import urls_in_text

HINT = "pip install 'local-deep-research[reports]'"

LABELS = {"en": ("Summary", "Sources", "Deep research report - local-deep-research"),
          "tr": ("Özet", "Kaynaklar", "Derin araştırma raporu — local-deep-research")}


def out_lang(root):
    try:
        import json
        x = json.loads((root / "harvest/profile.json").read_text(encoding="utf-8")).get("out_lang")
    except Exception:
        x = None
    return x if x in LABELS else "en"


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "research").resolve()
    report = (root / "REPORT.md").read_text(encoding="utf-8") if (root / "REPORT.md").exists() else ""
    urls = list(dict.fromkeys(urls_in_text(report)))

    wrote = [f for f in (write_xlsx(root, urls), write_pptx(root, report, urls)) if f]
    print(f"wrote {' and '.join(f.name for f in wrote) or 'nothing'} in {root}")


def write_xlsx(root, urls):
    try:
        import pandas as pd
        import openpyxl  # noqa: F401  the ExcelWriter engine
    except ImportError as e:
        print(f"REPORT.xlsx skipped: {e.name} is not installed ({HINT})", file=sys.stderr)
        return None
    xlsx = root / "REPORT.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for csv in sorted(root.glob("data/*.csv")):
            pd.read_csv(csv).to_excel(w, sheet_name=csv.stem[:31], index=False)
        pd.DataFrame({"#": range(1, len(urls) + 1), "URL": urls}).to_excel(w, sheet_name="Sources", index=False)
        cl = root / "claims/claims.csv"
        if cl.exists():  # every verified claim cluster (v3.1: nothing the harvest verified is left out)
            ev = pd.read_csv(cl, usecols=lambda c: c in ("confidence", "sources", "claims", "claim", "kind", "domains",
                                                         "urls", "quote", "anchored"))
            ev = ev[ev["confidence"] != "Düşük"]
            ev.to_excel(w, sheet_name="Evidence", index=False)
    return xlsx


def write_pptx(root, report, urls):
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError as e:
        print(f"REPORT.pptx skipped: {e.name} is not installed ({HINT})", file=sys.stderr)
        return None
    title = next((l[2:].strip() for l in report.splitlines() if l.startswith("# ")), "Research report")
    summary, grab = [], False
    for l in report.splitlines():
        if l.startswith("## "):
            if grab: break
            grab = True; continue
        if grab and l.strip().startswith(("-", "*")) and len(summary) < 7:
            summary.append(re.sub(r"[*_`]|\[(\d+)\]", "", l.strip()[1:].strip())[:160])

    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[0]); s.shapes.title.text = title
    s.placeholders[1].text = LABELS[out_lang(root)][2]
    if summary:
        s = prs.slides.add_slide(prs.slide_layouts[1]); s.shapes.title.text = LABELS[out_lang(root)][0]
        tf = s.placeholders[1].text_frame; tf.text = summary[0]
        for b in summary[1:]: tf.add_paragraph().text = b
        for p in tf.paragraphs: p.font.size = Pt(18)
    for fig in sorted(root.glob("fig/*.png")):
        s = prs.slides.add_slide(prs.slide_layouts[5]); s.shapes.title.text = fig.stem.replace("_", " ").replace("-", " ")
        s.shapes.add_picture(str(fig), Inches(1.2), Inches(1.4), height=Inches(5.6))
    s = prs.slides.add_slide(prs.slide_layouts[1]); s.shapes.title.text = LABELS[out_lang(root)][1]
    tf = s.placeholders[1].text_frame; tf.text = urls[0] if urls else "-"
    for u in urls[1:15]: tf.add_paragraph().text = u
    for p in tf.paragraphs: p.font.size = Pt(11)
    pptx = root / "REPORT.pptx"; prs.save(pptx)
    return pptx

if __name__ == "__main__":
    main()
