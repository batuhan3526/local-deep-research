"""Markdown -> PDF for research reports, or a self-contained HTML page when PDF is not available.
Usage: python -m local_deep_research.md2pdf research/REPORT.md [out.pdf]
Relative image paths (e.g. fig/chart.png) resolve against the markdown file's folder.

WeasyPrint needs the Pango libraries, which are often missing on Windows and macOS. Without it (or when it fails)
the same styled page is written as <name>.html with the images embedded, so it opens and prints from any browser.
Exit code 0 when a PDF or the HTML fallback was written; the last line printed says which."""
import base64
import html as _html
import mimetypes
import pathlib
import re
import sys

CSS_TEXT = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
body { font-family: 'DejaVu Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; line-height: 1.45;
       color: #1a1a1a; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13.5pt; margin-top: 18px; border-bottom: 1px solid #bbb; }
h3 { font-size: 11.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 4px 6px; vertical-align: top; }
th { background: #eceff3; }
tr:nth-child(even) td { background: #f8f9fa; }
img { max-width: 100%; display: block; margin: 8px auto; }
code { font-family: 'DejaVu Sans Mono', Consolas, Menlo, monospace; font-size: 8.5pt; background: #f1f1f1; padding: 0 2px; }
pre { background: #f1f1f1; padding: 6px; white-space: pre-wrap; font-size: 8.5pt; }
blockquote { border-left: 3px solid #999; margin-left: 0; padding-left: 10px; color: #444; }
a { color: #0b57d0; text-decoration: none; word-break: break-all; }
"""
SCREEN_CSS = "@media screen { body { max-width: 60em; margin: 2em auto; padding: 0 1em; } }"


def to_html(md_text):
    """Markdown -> HTML body (python-markdown when installed, else the text escaped in a <pre>)."""
    try:
        import markdown
    except ImportError:
        return f"<pre>{_html.escape(md_text)}</pre>"
    return markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"])


def inline_images(body, base):
    """Relative <img src> under `base` -> data: URIs, so the HTML file stands alone (moved, mailed, attached)."""
    base = base.resolve()

    def sub(m):
        src = _html.unescape(m.group(2))
        if re.match(r"(?i)^[a-z][a-z0-9+.-]*:", src):  # http:, data:, file:, a Windows drive: not relative
            return m.group(0)
        p = (base / src).resolve()
        if base not in p.parents or not p.is_file():
            return m.group(0)
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return f'{m.group(1)}data:{mime};base64,{base64.b64encode(p.read_bytes()).decode("ascii")}{m.group(3)}'
    return re.sub(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', sub, body)


def page(body, title):
    return (f"<!doctype html>\n<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>{_html.escape(title)}</title><style>{CSS_TEXT}{SCREEN_CSS}</style></head><body>{body}</body></html>\n")


def convert(src, out=None):
    """src .md -> the written file: out (default <src>.pdf), or <out>.html when WeasyPrint is missing or fails."""
    src = pathlib.Path(src).resolve()
    out = pathlib.Path(out).resolve() if out else src.with_suffix(".pdf")
    text = src.read_text(encoding="utf-8")
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), src.stem)
    body = to_html(text)
    try:
        from weasyprint import CSS, HTML
        HTML(string=f"<html><head><meta charset='utf-8'></head><body>{body}</body></html>",
             base_url=str(src.parent)).write_pdf(out, stylesheets=[CSS(string=CSS_TEXT)])
        return out
    except Exception as e:  # noqa: BLE001  ImportError, OSError (no Pango / cairo), or a rendering failure
        why = f"{type(e).__name__}: {str(e)[:200]}"
    html_out = out.with_suffix(".html")
    html_out.write_text(page(inline_images(body, src.parent), title), encoding="utf-8")
    print(f"PDF not available ({why}); wrote HTML instead. For PDF install WeasyPrint with Pango "
          f"(pip install 'local-deep-research[reports]', see the WeasyPrint install docs)", file=sys.stderr)
    return html_out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(__doc__)
    out = convert(argv[0], argv[1] if len(argv) > 1 else None)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
