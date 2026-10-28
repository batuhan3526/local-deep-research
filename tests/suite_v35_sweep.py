"""Offline checks for the v3.5 final sweep: corpus.corr_order (importance order with the usual order breaking ties),
corpus verify --priority reaching corroborate, harvest.sweep_priority (single-source Orta only, relevance x the
sub-question's need, kind bonus, off-topic / Düşük / 2-source left out), cmd_sweep (verify with the pair budget, no
early stop and the priority file; one pool-only round that searches nothing and selects from the carried pool;
idempotent, --again; no pool round without carried candidates), the cmd_run pool-only round and the pipeline phase.
No LLM, no network, no model: harvest.llm_json / search_all / corpus_cmd are faked, embeddings are deterministic.
usage: python tests/suite_v35_sweep.py [scratch_dir]"""
import argparse
import copy
import csv
import hashlib
import json
import re
import time

import numpy as np

from _suite import finish, scratch  # noqa: E402  (sets hermetic LDR_* env first)

SCRATCH = scratch("v35-sweep-")

from local_deep_research import corpus  # noqa: E402
from local_deep_research import dataset  # noqa: E402
from local_deep_research import harvest  # noqa: E402
from local_deep_research import pipeline  # noqa: E402
from local_deep_research import rt_embed  # noqa: E402

RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


def fake_embed(texts, model=None, device=None):  # "T<k>" -> axis k (+0.3 per "W<k>"); else hashed words, axes 32-95
    out = []
    for t in texts:
        v = np.zeros(96)
        ts = re.findall(r"\bT(\d+)\b", t)
        for k in ts:
            v[int(k)] += 1.0
        for k in re.findall(r"\bW(\d+)\b", t):
            v[int(k)] += 0.3
        if not ts:
            for w in re.findall(r"\w+", t.lower()):
                v[32 + int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 64] += 1
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out)


rt_embed.embed = corpus.embed = fake_embed
harvest.ce_scores = lambda pairs: None
harvest._sleep = lambda s: None
FIELDS = ["confidence", "sources", "claims", "claim", "kind", "domains", "urls", "quote"]

# ---------- 1. corpus.corr_order ----------
CL = [{"key": k, "head": {"claim": c}, "members": [0] * n} for k, c, n in
      (("a", "A", 1), ("b", "B", 5), ("c", "C", 3), ("d", "D", 1))]
own = {"a": True, "b": False, "c": False, "d": False}
check("corr_order: usual order = own source trusted first, then bigger",
      [c["key"] for c in corpus.corr_order(CL, own)] == ["a", "b", "c", "d"])
check("corr_order: with prio, score first (unscored last), the usual order breaking ties",
      [c["key"] for c in corpus.corr_order(CL, own, {"D": 0.9, "C": 0.5, "B": 0.5})] == ["d", "b", "c", "a"])
SEEN = {}
real_corr = corpus.corroborate


def spy_corr(st, k, max_pairs, conc, stop_window, stop_lift, prio=None):
    SEEN.update(max_pairs=max_pairs, stop_window=stop_window, prio=prio)
    return 0


corpus.corroborate = spy_corr
corpus.cluster_claims = lambda root: {"claims": [], "clusters": [], "out": root, "group": {}}
corpus.corr_results = lambda st: {}
corpus.confidences = lambda st, corr: []
corpus.write_claims = lambda st, corr: None
VR = SCRATCH / "verify"; VR.mkdir(parents=True)
(VR / "prio.json").write_text(json.dumps({"X": 1.0}), encoding="utf-8")
corpus.cmd_verify(argparse.Namespace(root=VR, max_pairs=50, k=6, concurrency=4, stop_window=0, stop_lift=0.03,
                                     priority=VR / "prio.json"))
check("corpus verify --priority: the file's scores reach corroborate with the pair budget and stop window",
      SEEN == {"max_pairs": 50, "stop_window": 0, "prio": {"X": 1.0}}, SEEN)
(VR / "bad.json").write_text("[1, 2]", encoding="utf-8")
corpus.cmd_verify(argparse.Namespace(root=VR, max_pairs=50, k=6, concurrency=4, stop_window=0, stop_lift=0.03,
                                     priority=VR / "bad.json"))
check("corpus verify --priority: a non-dict / unreadable file -> the usual order (prio None), no crash",
      SEEN["prio"] is None, SEEN)
corpus.corroborate = real_corr

# ---------- 2. sweep_priority ----------
SP = SCRATCH / "prio/research"; (SP / "claims").mkdir(parents=True); (SP / "harvest").mkdir()


def write_rows(root, rows):
    with open(root / "claims/claims.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        w.writerows({"confidence": c, "sources": n, "claims": 1, "claim": t, "kind": k, "domains": "a.org",
                     "urls": "https://a.org/x", "quote": "q"} for c, n, t, k in rows)


write_rows(SP, [("Yüksek", 2, f"T0 known fact {i}", "finding") for i in range(8)]       # S1 has 8 Yüksek
           + [("Orta", 1, "T0 single one", "finding"), ("Orta", 1, "T1 single two", "finding"),
              ("Orta", 1, "T1 single three", "number"), ("Orta", 2, "T1 two sources", "finding"),
              ("Düşük", 1, "T1 low", "finding"), ("Orta", 1, "unrelated words entirely", "finding"),
              ("Orta", 0, "T1 W0 leaning", "limitation")])
subs = [{"id": 1, "title": "a", "text": "T0 first"}, {"id": 2, "title": "b", "text": "T1 second"}]
pr = harvest.sweep_priority(SP, subs)
check("sweep_priority: only single-source (0/1) Orta clusters near a sub-question; Düşük, 2-source and off-topic out",
      set(pr) == {"T0 single one", "T1 single two", "T1 single three", "T1 W0 leaning"}, pr)
check("sweep_priority: the sub-question with no Yüksek (need 1) ranks above the one with 8 (need 1/3)",
      pr["T1 single two"] == 1.0 and abs(pr["T0 single one"] - 1 / 3) < 1e-3, pr)
check("sweep_priority: a number / limitation claim gets SWEEP_KIND_BONUS",
      abs(pr["T1 single three"] - pr["T1 single two"] - harvest.SWEEP_KIND_BONUS) < 1e-9
      and pr["T1 W0 leaning"] > pr["T0 single one"], pr)
check("sweep_priority: no claims / no sub-questions -> {}",
      harvest.sweep_priority(SCRATCH / "none", subs) == {} and harvest.sweep_priority(SP, []) == {})


# ---------- 3. cmd_sweep + the pool-only round (cmd_run with fakes) ----------
class Clock:
    now = 1e9

    def time(self):
        return Clock.now

    def __getattr__(self, k):
        return getattr(time, k)


harvest.time = Clock()
harvest.interim_report = lambda *a, **k: None
dataset.export = lambda root: None
RUN = {"root": None, "searched": [], "corpus": []}
RAW = {"subquestions": [{"id": 1, "title": "Purged", "text": "T0 purged cross validation leakage",
                         "queries": [{"q": f"purged q{i}", "kind": "academic"} for i in range(6)]},
                        {"id": 2, "title": "Target", "text": "T1 etf direction target design",
                         "queries": [{"q": f"target q{i}", "kind": "practitioner"} for i in range(6)]}]}


def run_llm(prompt, max_tokens=6000, system=None):
    if "Split the research" in prompt:
        return copy.deepcopy(RAW)
    if "quality checks" in prompt:
        return {"subquestions": [{"id": 1, "checks": ["definitive"]}, {"id": 2, "checks": ["definitive"]}]}
    return {"gaps": ["x"], "queries": [{"q": f"gap {i}", "kind": "academic"} for i in range(6)]}


def run_search(queries, grace=None, cap=None):
    RUN["searched"].append(list(queries))
    n = len(RUN["searched"])
    return [{"url": f"https://s{n}-{j}-{i}.org/p", "title": f"T{qq[0] - 1} page {qq[1]} {i}", "snippet": "x",
             "year": "", "src": "web", "sub": qq[0], "query": qq[1]} for j, qq in enumerate(queries) for i in range(3)]


def cur_round():
    return json.loads((RUN["root"] / "harvest/state.json").read_text(encoding="utf-8"))["round"]


def run_corpus(*args):
    RUN["corpus"].append(args)
    if args[0] == "extract":
        Clock.now += 300
        return
    Clock.now += 120
    r = cur_round()
    write_rows(RUN["root"], [("Yüksek", 2, f"T0 fact {j} r{r}", "finding") for j in range(5 * r)]
               + [("Orta", 1, f"T1 single {j} r{r}", "finding") for j in range(4)])
    (RUN["root"] / "claims/.verify_stats.json").write_text(json.dumps(
        {"claims": 9, "grounded": 9, "corr_s": 0, "verify_s": 0, "judged": 77, "lifted": 3, "max_pairs": 100,
         "early_stop": False}), encoding="utf-8")


harvest.llm_json, harvest.search_all, harvest.corpus_cmd = run_llm, run_search, run_corpus
corpus.fetch_one = lambda u, *a: (u, 200, "purged cross validation leakage etf direction " * 40)


def new_run(name, **kw):
    root = SCRATCH / f"{name}/research"; (root / "claims").mkdir(parents=True); (root / "harvest").mkdir()
    (root / "BRIEF.md").write_text("# T\n\n**Soru:** machine learning for ETF direction?\n", encoding="utf-8")
    (root / "harvest/.concurrency").write_text("8\n", encoding="utf-8")
    RUN.update(root=root, searched=[], corpus=[])
    base = dict(root=root, rounds=1, subs=2, per_sub=6, pages=6, concurrency=8, max_pairs=100, min_yield_ratio=0.35,
                early_stop=False, stop_on_pass=False, until=1, patience=1, no_emerge=True, no_probe=True)
    return argparse.Namespace(**{**base, **kw})


def ri(root, k):
    return json.loads((root / f"harvest/round-{k}.json").read_text(encoding="utf-8"))


H = new_run("sw")
harvest.cmd_run(H)
root = RUN["root"]
r1 = ri(root, 1)
carry = {c["url"] for c in r1.get("carry") or []}
check("setup: round 1 left a carry pool (more good candidates than its 6 pages)", len(carry) > 6, len(carry))
RUN["searched"].clear(); RUN["corpus"].clear()
S = argparse.Namespace(root=root, rounds=None, subs=2, per_sub=6, pages=6, concurrency=8, max_pairs=100,
                       min_yield_ratio=0.35, early_stop=False, stop_on_pass=False, patience=1, again=False,
                       pairs=1234, no_emerge=False, no_probe=False)
harvest.cmd_sweep(S)
vcall = next((c for c in RUN["corpus"] if c[0] == "verify"), ())
check("sweep verify: corpus verify with the pair budget, no early stop (--stop-window 0) and the priority file",
      "--max-pairs" in vcall and vcall[vcall.index("--max-pairs") + 1] == "1234"
      and vcall[vcall.index("--stop-window") + 1] == "0"
      and vcall[vcall.index("--priority") + 1].endswith("claims/.sweep_priority.json"), vcall)
pf = json.loads((root / "claims/.sweep_priority.json").read_text(encoding="utf-8"))
check("sweep verify: the priority file holds the single-source clusters (sweep_priority)",
      set(pf) == {f"T1 single {j} r1" for j in range(4)}, pf)
sw = json.loads((root / "harvest/sweep.json").read_text(encoding="utf-8"))
r2 = ri(root, 2)
check("sweep pool round: round 2 ran, searched nothing, selected only carried pages",
      all(q == [] for q in RUN["searched"]) and r2["queries"] == 0 and r2["sel"]
      and all(c["url"] in carry for c in r2["sel"] if c.get("src") != "link"), (RUN["searched"], r2.get("sel")))
check("sweep: harvest/sweep.json records the verify part (judged, lifted, yields) and the pool round",
      sw["verify"]["judged"] == 77 and sw["verify"]["lifted"] == 3 and "yield_after" in sw["verify"]
      and sw["pool"]["round"] == 2 and sw["pool"]["pages"] == len(r2["sel"]), sw)
r1b = ri(root, 1)  # emerge / probe for a round are stored in the finished round before it
check("sweep pool round: no emerge / probe though the caller allowed them (a pool round does not grow the plan)",
      "emerge" not in r1b and "probe" not in r1b and S.no_emerge is True and S.no_probe is True, list(r1b))
n_corpus = len(RUN["corpus"])
n_searched = len(RUN["searched"])
harvest.cmd_sweep(S)
check("sweep: idempotent (a second run does nothing)", len(RUN["corpus"]) == n_corpus
      and len(RUN["searched"]) == n_searched
      and harvest.load_state(root)["round"] == 2)
S.again = True
harvest.cmd_sweep(S)
check("sweep --again: both parts again (verify call + pool round 3)",
      len([c for c in RUN["corpus"][n_corpus:] if c[0] == "verify"]) >= 2 and harvest.load_state(root)["round"] == 3)
H2 = new_run("sw-empty")
harvest.cmd_run(H2)
r = ri(RUN["root"], 1)
r["carry"], r["links"] = [], []
harvest.round_file(RUN["root"], 1).write_text(json.dumps(r), encoding="utf-8")
S2 = argparse.Namespace(**{**vars(S), "root": RUN["root"], "again": False})
harvest.cmd_sweep(S2)
sw2 = json.loads((RUN["root"] / "harvest/sweep.json").read_text(encoding="utf-8"))
check("sweep: no carried candidates -> verify part only, pool recorded as skipped, no new round",
      "verify" in sw2 and sw2["pool"]["round"] is None and harvest.load_state(RUN["root"])["round"] == 1, sw2)

# ---------- 4. pipeline phase ----------
names = [p.name for p in pipeline.phases()]
check("pipeline: sweep right after harvest, before contradictions / notes / gap",
      names.index("harvest") + 1 == names.index("sweep") < names.index("notes") < names.index("gap"), names)
D = SCRATCH / "phase/research"; (D / "harvest").mkdir(parents=True)
check("pipeline: sweep not done without sweep.json; done with both parts, with .sweep_done, or once analysis started",
      not pipeline._sweep_done(D)
      and ((D / "harvest/sweep.json").write_text(json.dumps({"verify": {}}), encoding="utf-8") or True)
      and not pipeline._sweep_done(D)
      and ((D / "harvest/sweep.json").write_text(json.dumps({"verify": {}, "pool": {}}), encoding="utf-8") or True)
      and pipeline._sweep_done(D))
from local_deep_research import stages  # noqa: E402
check("stages: the sweep runs with the collecting model", stages.stage_of("sweep", names) == "collect")
harvest.time = time

finish(RESULTS, SCRATCH)
