"""Archive every source URL cited in a research folder as plain text, without spending LLM tokens.
Usage: python -m local_deep_research.archive_sources research/
Reads URLs from notes/, leads/, LEADS.md, ANALYSIS.md, REPORT.md; writes sources/NNN-<slug>.md and sources/INDEX.md."""
import re, sys, pathlib, datetime, concurrent.futures as cf
import httpx, trafilatura

from .srcclass import URL_RE  # balanced parentheses stay in the URL
UA = {"User-Agent": "Mozilla/5.0 (research-archiver; local agent)"}

def collect(root: pathlib.Path):
    files = list(root.glob("notes/*.md")) + list(root.glob("leads/*.md")) + \
            [root / n for n in ("LEADS.md", "ANALYSIS.md", "REPORT.md") if (root / n).exists()]
    urls = []
    for f in files:
        for u in URL_RE.findall(f.read_text(encoding="utf-8", errors="replace")):
            u = u.rstrip(".;:")
            if u not in urls:
                urls.append(u)
    return urls

def fetch(u):
    # same guarded fetcher as corpus.py (public hosts only, size/time caps, sandboxed PDF parsing); lazy import
    # because corpus imports collect() from here
    from . import corpus
    return corpus.fetch_one(u)

def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "research").resolve()
    out = root / "sources"; out.mkdir(exist_ok=True)
    urls = collect(root)
    index = [f"# Archived sources ({datetime.date.today()})", "", "| # | Status | Chars | File | URL |", "|---|---|---|---|---|"]
    with cf.ThreadPoolExecutor(8) as ex:
        for i, (u, st, text) in enumerate(ex.map(fetch, urls), 1):
            slug = re.sub(r"[^a-z0-9]+", "-", u.split("//", 1)[-1].lower())[:60].strip("-")
            name = f"{i:03d}-{slug}.md"
            (out / name).write_text(f"Source: {u}\nFetched: {datetime.datetime.now():%Y-%m-%d %H:%M}\nHTTP: {st}\n\n{text}\n", encoding="utf-8")
            index.append(f"| {i} | {st} | {len(text)} | {name} | {u} |")
    (out / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    ok = sum(1 for l in index[4:] if "| 200 |" in l)
    print(f"archived {len(urls)} urls ({ok} HTTP 200) -> {out}")

if __name__ == "__main__":
    main()
