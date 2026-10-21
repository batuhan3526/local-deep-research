"""Offline checks for the scholarly preference in harvest.select: up to ACADEMIC_SHARE of a round's pages go to
relevant scholarly candidates (journals / DOI / scholarly encyclopedias / university repositories), the rest by score;
a topic with few scholarly hits leaves the share to the rest; irrelevant scholarly pages are not forced in.
Embeddings and the cross-encoder are faked (no model, no network).
usage: python tests/suite_academic.py"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from local_deep_research import harvest, rt_embed, srcclass  # noqa: E402

RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


REL = {}  # text -> relevance the fake embedding gives against the sub-question


def fake_embed(texts, model=None, device=None):
    out = []
    for t in texts:
        r = REL.get(t.split(".", 1)[0], 1.0)  # title before the first "."; the sub-question -> [1, 0], so cosine = r
        out.append(np.array([r, (1 - r * r) ** 0.5]))
    return np.array(out)


rt_embed.embed = fake_embed
harvest.ce_scores = lambda pairs, model=None: [0.5] * len(pairs)
SUB = [{"id": 1, "text": "Why did the siege fail"}]


def cand(url, title, rel):
    REL[title] = rel
    return {"url": url, "title": title, "snippet": "s", "query": "q", "src": "web", "sub": 1, "year": ""}


blogs = [cand(f"https://historysite{i}.com/articles/siege", f"blog{i}", 0.80) for i in range(30)]  # distinct domains: no cap
papers = [cand(f"https://doi.org/10.1000/p{i}", f"paper{i}", 0.62) for i in range(30)]
repos = [cand(f"https://docs.lib.purdue.edu/etd/{i}/viewcontent/t{i}.pdf", f"thesis{i}", 0.66) for i in range(5)]
off = [cand(f"https://doi.org/10.1000/off{i}", f"offtopic{i}", 0.05) for i in range(10)]

harvest.ACADEMIC_SHARE = 0.4
sel = harvest.select(blogs + papers + repos + off, SUB, set(), 20)
n_s = sum(srcclass.scholarly(c["url"]) for c in sel)
check("plenty of scholarly candidates: >= 40% of 20 pages are scholarly although blogs score higher",
      len(sel) == 20 and n_s >= 8, (len(sel), n_s))
check("irrelevant scholarly pages are never forced in (relevance floor still applies)",
      not any("offtopic" in c["title"] for c in sel))
check("university repository PDFs count as scholarly", any("thesis" in c["title"] for c in sel)
      and srcclass.scholarly(repos[0]["url"]) and not srcclass.academic(repos[0]["url"]))

harvest.ACADEMIC_SHARE = 0.0
sel0 = harvest.select(blogs + papers + repos + off, SUB, set(), 20)
check("share 0 = old behaviour: pure score order picks the blogs",
      sum(srcclass.scholarly(c["url"]) for c in sel0) < n_s, sum(srcclass.scholarly(c["url"]) for c in sel0))

harvest.ACADEMIC_SHARE = 0.4
few = harvest.select(blogs + papers[:2], SUB, set(), 20)
check("few scholarly candidates: both taken, the rest of the share goes to other pages (20 still selected)",
      len(few) == 20 and sum(srcclass.scholarly(c["url"]) for c in few) == 2, len(few))
check("scholarly hosts: Belleten / TDV İA academic; a university magazine page and Wikipedia are not scholarly",
      srcclass.academic("https://belleten.gov.tr/tam-metin/1/tur") and srcclass.academic("https://islamansiklopedisi.org.tr/x")
      and not srcclass.scholarly("https://reed.edu/reed-magazine/articles/2015/x.html")
      and not srcclass.scholarly("https://en.wikipedia.org/wiki/X"))

print(f"\n{RESULTS[0]} passed, {RESULTS[1]} failed")
sys.exit(1 if RESULTS[1] else 0)
