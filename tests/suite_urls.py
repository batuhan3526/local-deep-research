"""Offline checks for URL extraction in report text (srcclass.URL_RE / urls_in_text, used by sections, verify_fix,
appendix, export_extras, archive_sources). Live run 2026-10-20: the old pattern stopped at ")", so URLs such as
Wikipedia's "Polish–Ottoman_War_(1683–1699)" were cut and the code-built sources lists carried 404 links.
usage: python tests/suite_urls.py"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from local_deep_research import sections, srcclass  # noqa: E402

RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


W = "https://en.wikipedia.org/wiki/Polish%E2%80%93Ottoman_War_(1683%E2%80%931699)"
t = f"see {W}. Also (https://a.org/x) and [w](https://b.org/c_(d)), then https://c.org/y⟨S1-0001⟩ and https://d.org/z;"
u = srcclass.urls_in_text(t)
check("balanced (...) stays in the URL; a wrapping ')' / '.' / ';' / '⟨id⟩' does not",
      u == [W, "https://a.org/x", "https://b.org/c_(d)", "https://c.org/y", "https://d.org/z"], u)
src = sections.sources_list(t, "English")
check("sources list keeps the full Wikipedia URL (no cut '(1683…1699' entry)", f"1. {W}\n" in src and "(1683%E2%80%931699\n" not in src,
      src)
kept = sections.keep_urls(t, {W, "https://a.org/x"})[0]
check("keep_urls: allowed parenthesised URL kept whole, the rest removed, wrapping parens untouched",
      W in kept and "(https://a.org/x)" in kept and "b.org" not in kept and "c.org" not in kept, kept)

print(f"\n{RESULTS[0]} passed, {RESULTS[1]} failed")
sys.exit(1 if RESULTS[1] else 0)
