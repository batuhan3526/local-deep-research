"""Offline checks for v3.4 novelty (paraphrase-duplicate clusters are not new information): claim_vectors (sha1-keyed
float16 cache, incremental, CPU model), novelty (new / novel / duplicate counts, NOVEL_SIM edge, Düşük and unknown-URL
clusters left out, upgrades of older clusters never duplicates, per-sub-question duplicate yield like packs(), newborn
sub-questions only from their round), apply_novelty, ref_ypm / sub_hist / plateaus on adjusted history with fallback,
backfill_novelty (idempotent, skips unfinished rounds) and the cmd_run wiring (weak rule on the adjusted yield/min,
round json fields, ROUNDS.md "yenilik", automatic backfill of an old run).
No LLM, no network, no model: harvest.llm_json is faked, embeddings are deterministic vectors.
usage: python tests/suite_v34_novelty.py [scratch_dir]"""
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import re
import time

import numpy as np

from _suite import finish, scratch  # noqa: E402  (sets hermetic LDR_* env first)

SCRATCH = scratch("v34-test-")

from local_deep_research import corpus  # noqa: E402
from local_deep_research import dataset  # noqa: E402
from local_deep_research import harvest  # noqa: E402
from local_deep_research import rt_embed  # noqa: E402

RESULTS = [0, 0]
print("harvest module:", harvest.__file__)


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


# ---------- deterministic embeddings ----------
# "T<k>" = unit basis vector k; "V<k>@<c>" = cosine c to basis k, the rest in a dimension of its own (per text);
# anything else: a dimension of its own (orthogonal to everything).
DIM, OWN = 2048, {}
EMB_CALLS = []


def fake_embed(texts, model=None, device=None):
    EMB_CALLS.append((list(texts), model, device))
    out = []
    for t in texts:
        v = np.zeros(DIM)
        own = OWN.setdefault(t, 200 + len(OWN))
        m = re.search(r"\bV(\d+)@([\d.]+)", t)
        ts = re.findall(r"\bT(\d+)\b", t)
        if m:
            c = float(m.group(2))
            v[int(m.group(1))] = c
            v[own] = math.sqrt(max(0.0, 1 - c * c))
        elif ts:
            for k in ts:
                v[int(k)] += 1.0
        else:
            v[own] = 1.0
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out)


rt_embed.embed = corpus.embed = fake_embed
harvest.ce_scores = lambda pairs: None
harvest._sleep = lambda s: None
FIELDS = ["confidence", "sources", "claims", "claim", "kind", "domains", "urls", "quote"]


def cl(claim, conf, urls):
    return {"confidence": conf, "sources": 2, "claims": 1, "claim": claim, "kind": "finding",
            "domains": "a.example, b.example", "urls": urls, "quote": "q"}


def write_claims(root, rows):
    (root / "claims").mkdir(parents=True, exist_ok=True)
    with open(root / "claims/claims.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def write_sources(root, urls_by_round):
    (root / "harvest").mkdir(parents=True, exist_ok=True)
    with open(root / "harvest/sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["round", "sub", "score", "src", "year", "url", "title", "query", "lang"])
        w.writeheader()
        for rnd, us in urls_by_round.items():
            for u in us:
                w.writerow({"round": rnd, "sub": 1, "score": 1, "src": "web", "url": u})


def write_round(root, k, **info):
    (root / "harvest").mkdir(parents=True, exist_ok=True)
    harvest.round_file(root, k).write_text(json.dumps(info), encoding="utf-8")


U1, U2, U3 = "https://r1.example/x", "https://r2.example/x", "https://r3.example/x"
ROWS = [cl("T0 old tools", "Yüksek", U1), cl("T1 old method", "Yüksek", U1), cl("T2 old untrusted", "Orta", U1),
        cl("T3 old low", "Düşük", U1), cl("T5 old far", "Yüksek", U1),
        cl("T4 upgraded", "Yüksek", f"{U1} {U2}"),            # first seen in round 1, corroborated in round 2
        cl("V0@0.95 tools again", "Yüksek", U2),             # dup of T0 (S1)
        cl("V1@0.90 method again", "Yüksek", U2),            # dup of T1 (S2)
        cl("V1@0.95 method untrusted", "Orta", U2),           # dup of T1, yield 0
        cl("V0@0.50 new angle", "Yüksek", U2),               # novel
        cl("V3@0.99 echo of a Düşük", "Yüksek", U2),         # only near a Düşük cluster: novel
        cl("V0@0.855 just above", "Yüksek", U2),             # dup (edge)
        cl("V1@0.845 just below", "Yüksek", U2),             # novel (edge)
        cl("V5@0.90 far dup", "Yüksek", U2),                 # dup of T5, near no sub-question
        cl("V0@0.99 low new", "Düşük", U2),                  # Düşük: not counted
        cl("V0@0.99 unknown url", "Yüksek", "https://zzz.example/y"),  # URL not in sources.csv: no round
        cl("V2@0.90 r3 dup of untrusted", "Yüksek", U3)]     # round 3: dup of T2 -> S3 (born 3)
SUBS = [{"id": 1, "text": "T0 tools"}, {"id": 2, "text": "T1 methods"}, {"id": 3, "text": "T2 newborn", "born": 3}]
check("NOVEL_SIM is a probability-like cosine between 0.7 and 0.97", 0.7 <= harvest.NOVEL_SIM <= 0.97, harvest.NOVEL_SIM)
OLD_SIM = harvest.NOVEL_SIM
harvest.NOVEL_SIM = 0.85

# ---------- 1. novelty counts ----------
R = SCRATCH / "nov/research"
write_sources(R, {1: [U1], 2: [U2], 3: [U3]})
write_claims(R, ROWS)
EMB_CALLS.clear()
n2 = harvest.novelty(R, SUBS, 2)
check("novelty r2: 8 new (Düşük / unknown-URL / upgraded older cluster left out), 3 novel, 5 paraphrase duplicates",
      n2["new"] == 8 and n2["novel"] == 3 and n2["share"] == round(3 / 8, 4) and n2["sim"] == 0.85, n2)
check("novelty r2: dup_yield = 1 + 1 + 0 (untrusted Orta) + 1 (edge) + 1 (far) = 4; the upgraded T4 is not a duplicate",
      n2["dup_yield"] == 4.0, n2)
check("novelty r2: per-sub duplicate yield S1 2 (0.95, 0.855), S2 1 (0.90 + untrusted 0); the far one in no sub; "
      "S3 (born 3) absent", n2["sub_dup_yield"] == {"1": 2.0, "2": 1.0}, n2["sub_dup_yield"])
check("novelty: embeddings asked of the CPU model on device cpu (the GPUs serve the LLM)",
      EMB_CALLS and all(m == rt_embed.CPU_MODEL and d == "cpu" for _, m, d in EMB_CALLS),
      [(m, d) for _, m, d in EMB_CALLS])
n1 = harvest.novelty(R, SUBS, 1)
check("novelty r1: nothing older -> every new cluster novel, share 1.0, dup 0",
      n1["new"] == 5 and n1["novel"] == 5 and n1["share"] == 1.0 and n1["dup_yield"] == 0, n1)
n3 = harvest.novelty(R, SUBS, 3)
check("novelty r3: the newborn S3 (born 3) gets its duplicate's yield",
      n3["new"] == 1 and n3["novel"] == 0 and n3["sub_dup_yield"] == {"1": 0.0, "2": 0.0, "3": 1.0}, n3)
n9 = harvest.novelty(R, SUBS, 9)
check("novelty for a round with nothing new: new 0, share None, dup 0",
      n9["new"] == 0 and n9["share"] is None and n9["dup_yield"] == 0, n9)
check("novelty threshold edge: sim 0.86 -> the 0.855 paraphrase counts as novel; sim 0.84 -> the 0.845 one is a dup",
      harvest.novelty(R, SUBS, 2, sim=0.86)["novel"] == 4 and harvest.novelty(R, SUBS, 2, sim=0.84)["novel"] == 2)
check("novelty: no claims / sources yet -> None", harvest.novelty(SCRATCH / "empty", SUBS, 2) is None)

# ---------- 2. embedding cache ----------
cf = R / harvest.CLAIM_EMB
z = np.load(cf)
check("cache: harvest/.claim_emb.npz holds one float16 vector per distinct grounded claim, keyed by sha1, model name",
      z["vecs"].dtype == np.float16 and len(z["keys"]) == 14 and len(z["keys"][0]) == 40
      and str(z["model"]) == rt_embed.CPU_MODEL, (z["vecs"].dtype, len(z["keys"])))
EMB_CALLS.clear()
harvest.novelty_data(R)
check("cache: a second read embeds nothing", EMB_CALLS == [], EMB_CALLS)
write_claims(R, ROWS + [cl("T7 brand new", "Yüksek", U3)])
d = harvest.novelty_data(R)
check("cache incremental: one new claim -> only that text is embedded",
      [c[0] for c in EMB_CALLS] == [["T7 brand new"]] and d["embedded"] == 1 and len(np.load(cf)["keys"]) == 15,
      [c[0] for c in EMB_CALLS])
np.savez(cf, keys=np.load(cf)["keys"], vecs=np.load(cf)["vecs"], model=np.array("another-model"))
EMB_CALLS.clear()
harvest.novelty_data(R)
check("cache built by another model -> rebuilt (all 15 embedded again)", sum(len(c[0]) for c in EMB_CALLS) == 15,
      sum(len(c[0]) for c in EMB_CALLS))
tag = hashlib.sha1(rt_embed.CPU_MODEL.encode()).hexdigest()[:8]
seed_txt = "T8 seeded by verify"
np.save(R / f"claims/.claimvec.{tag}.npy", fake_embed(["T9 other", seed_txt]).astype(np.float32))
(R / f"claims/.claimvec.{tag}.keys").write_text("\n".join(hashlib.sha1(t.encode()).hexdigest()[:20]
                                                          for t in ("T9 other", seed_txt)), encoding="utf-8")
write_claims(R, ROWS + [cl("T7 brand new", "Yüksek", U3), cl(seed_txt, "Yüksek", U3)])
EMB_CALLS.clear()
d = harvest.novelty_data(R)
check("cache seeded from verify's claims/.claimvec.<model>.npy: a claim found there is not embedded, its vector used",
      EMB_CALLS == [] and d["embedded"] == 0 and np.argmax(d["v"][[r["claim"] for r in d["rows"]].index(seed_txt)]) == 8,
      (EMB_CALLS, d["embedded"]))
alt = SCRATCH / "alt.npz"
harvest.novelty_data(R, cache=alt)
check("cache=<file>: written there (read-only experiments on a live run dir)", alt.exists())
write_claims(R, ROWS)

# ---------- 3. apply_novelty, ref_ypm / sub_hist / plateaus on adjusted history ----------
info = harvest.apply_novelty({"new_yield": 10.0, "minutes": 2.0, "sub_new": {"1": 5.0, "2": 5.0}},
                             {"new": 8, "novel": 3, "share": 0.375, "dup_yield": 4.0,
                              "sub_dup_yield": {"1": 2.0, "2": 1.0}, "sim": 0.85})
check("apply_novelty: new_yield_adj 6, ypm_adj 3.0, sub_new_adj S1 3 / S2 4, novelty stored",
      info["new_yield_adj"] == 6.0 and info["ypm_adj"] == 3.0 and info["sub_new_adj"] == {"1": 3.0, "2": 4.0}
      and info["novelty"]["novel"] == 3, info)
H1 = SCRATCH / "hist/research"
write_round(H1, 1, ypm=10.0, ypm_adj=4.0)
write_round(H1, 2, ypm=3.0, ypm_adj=3.5)
check("ref_ypm: the adjusted values when present (max(4.0, 3.5) = 4.0, not round 1's raw 10.0)",
      harvest.ref_ypm(H1, 3) == 4.0, harvest.ref_ypm(H1, 3))
write_round(H1, 2, ypm=6.0)
check("ref_ypm fallback: a round without ypm_adj counts with its ypm (max(4.0, 6.0) = 6.0)",
      harvest.ref_ypm(H1, 3) == 6.0, harvest.ref_ypm(H1, 3))
write_round(H1, 2, ypm=6.0, ypm_adj=0.0)
check("ref_ypm: an adjusted 0.0 is a value, not 'missing' (max(4.0, 0.0) = 4.0)", harvest.ref_ypm(H1, 3) == 4.0,
      harvest.ref_ypm(H1, 3))
H2 = SCRATCH / "plat/research"
write_round(H2, 1, sub_new={"1": 10, "2": 10}, sub_new_adj={"1": 10, "2": 10})
write_round(H2, 2, sub_new={"1": 9, "2": 9}, sub_new_adj={"1": 2, "2": 9})
write_round(H2, 3, sub_new={"1": 9, "2": 9}, sub_new_adj={"1": 2, "2": 9})
ps = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
check("sub_hist reads sub_new_adj when present", harvest.sub_hist(H2, ps[0], 3) == [10, 2, 2], harvest.sub_hist(H2, ps[0], 3))
check("plateaus on adjusted history: S1 (raw 10, 9, 9 = not flat; adjusted 10, 2, 2) plateaus, S2 does not",
      harvest.plateaus(H2, ps, 3) == {1: 3}, harvest.plateaus(H2, ps, 3))
write_round(H2, 3, sub_new={"1": 2, "2": 9})
check("sub_hist fallback: a round without sub_new_adj reads its sub_new", harvest.sub_hist(H2, ps[0], 3) == [10, 2, 2],
      harvest.sub_hist(H2, ps[0], 3))

# ---------- 4. backfill ----------
for k, (ny, mn) in {1: (8.0, 4.0), 2: (10.0, 2.0)}.items():
    write_round(R, k, new_yield=ny, minutes=mn, ypm=round(ny / mn, 3), sub_new={"1": ny / 2, "2": ny / 2})
write_round(R, 3, sel=[], t={"search": 1.0})  # round 3 in progress (no verify yet): skipped
out = io.StringIO()
with contextlib.redirect_stdout(out):
    b1 = harvest.backfill_novelty(R, SUBS)
r2 = harvest.round_info(R, 2)
check("backfill: finished rounds 1-2 get novelty / ypm_adj / sub_new_adj, round 3 in progress untouched",
      sorted(b1) == [1, 2] and r2["ypm_adj"] == 3.0 and r2["new_yield_adj"] == 6.0
      and r2["sub_new_adj"] == {"1": 3.0, "2": 4.0} and harvest.round_info(R, 1)["ypm_adj"] == 2.0
      and "ypm_adj" not in harvest.round_info(R, 3), (b1, r2))
check("backfill logs one line per round with the novelty and the yield/min change",
      "round 2: novelty: 3/8 new clusters novel (38%), paraphrase-duplicate yield -4; yield/min 5.0 -> 3.0" in out.getvalue(),
      out.getvalue())
before = harvest.round_file(R, 2).read_bytes()
EMB_CALLS.clear()
check("backfill re-run: nothing to do, no embedding, files unchanged",
      harvest.backfill_novelty(R, SUBS, log=lambda *a, **k: None) == {} and EMB_CALLS == []
      and harvest.round_file(R, 2).read_bytes() == before)
harvest.backfill_novelty(R, SUBS, force=True, log=lambda *a, **k: None)
check("backfill --force: same values (deterministic)", harvest.round_info(R, 2) == json.loads(before))
write_round(R, 1, new_yield=8.0, minutes=4.0, ypm=2.0, sub_new={"1": 4.0, "2": 4.0})
harvest.backfill_novelty(R, SUBS, write=False, log=lambda *a, **k: None)
check("backfill write=False (--dry-run): round files untouched", "ypm_adj" not in harvest.round_info(R, 1))

# ---------- 5. cmd_run: weak rule on the adjusted yield/min, round json, ROUNDS.md, automatic backfill ----------
class Clock:
    now = 1e9

    def time(self):
        return Clock.now

    def __getattr__(self, k):
        return getattr(time, k)


harvest.time = Clock()
harvest.interim_report = lambda *a, **k: None
dataset.export = lambda root: None
harvest.emerge = lambda root, pl, rnd, a: {"candidates": 0, "born": [], "why": "test"}
RUN = {"root": None, "rows": None}
RAW = {"subquestions": [{"id": 1, "title": "Tools", "text": "T0 osint tools",
                         "queries": [{"q": f"tools q{i}", "kind": "academic"} for i in range(6)]},
                        {"id": 2, "title": "Methods", "text": "T1 osint methods",
                         "queries": [{"q": f"methods q{i}", "kind": "practitioner"} for i in range(6)]}]}
GQ = [0]


def run_llm(prompt, max_tokens=6000, system=None):
    if "Split the research" in prompt:
        return copy.deepcopy(RAW)
    if "quality checks" in prompt:
        return {"subquestions": [{"id": 1, "checks": ["definitive"]}, {"id": 2, "checks": ["definitive"]}]}
    GQ[0] += 1
    return {"gaps": ["x"], "queries": [{"q": f"gap {GQ[0]} {i}", "kind": "academic"} for i in range(12)]}


def run_search(queries):
    return [{"url": f"https://s{GQ[0]}-{j}.org/p", "title": f"page {qq[1]}", "snippet": f"{qq[1]} " * 5, "year": "",
             "src": "web", "sub": qq[0], "query": qq[1]} for j, qq in enumerate(queries)]


def cur_round():
    return json.loads((RUN["root"] / "harvest/state.json").read_text(encoding="utf-8"))["round"]


def run_corpus(*args):
    rnd, root = cur_round(), RUN["root"]
    if args[0] == "extract":  # this round's claim URL joins sources.csv (fetch wrote the round's pages)
        Clock.now += 10 * 60
        with open(root / "harvest/sources.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rnd, 1, 1, "web", "", f"https://r{rnd}.example/x", "", "", ""])
        return
    rows = RUN["rows"](rnd)
    write_claims(root, rows)
    (root / "claims/.verify_stats.json").write_text(json.dumps(
        {"claims": len(rows), "grounded": len(rows), "corr_s": 0, "verify_s": 0, "judged": 100, "max_pairs": 100,
         "early_stop": False}), encoding="utf-8")


harvest.search_all, harvest.corpus_cmd = run_search, run_corpus
corpus.fetch_one = lambda u, *a: (u, 200, "osint tools methods " * 40)


def fresh(r, k0, n):  # n Yüksek clusters on their own basis vectors (orthogonal to everything: novel)
    return [cl(f"T{k} fact", "Yüksek", f"https://r{r}.example/x") for k in range(k0, k0 + n)]


def para(r, k0, n):  # n Yüksek paraphrases (cos 0.95) of fresh(1, k0, n)
    return [cl(f"V{k}@0.95 fact again", "Yüksek", f"https://r{r}.example/x") for k in range(k0, k0 + n)]


ROUND_ROWS = {1: fresh(1, 10, 20), 2: fresh(2, 30, 20), 3: para(3, 10, 20)}
rows_d = lambda r: [x for k in range(1, r + 1) for x in ROUND_ROWS[k]]  # noqa: E731


def new_run(name, rows, **kw):
    root = SCRATCH / f"{name}/research"; (root / "claims").mkdir(parents=True); (root / "harvest").mkdir()
    (root / "BRIEF.md").write_text("# T\n\n**Soru:** OSINT methods?\n", encoding="utf-8")
    (root / "harvest/.concurrency").write_text("8\n", encoding="utf-8")
    RUN.update(root=root, rows=rows)
    harvest.llm_json = run_llm
    base = dict(root=root, rounds=6, subs=2, per_sub=6, pages=20, concurrency=8, max_pairs=100, min_yield_ratio=0.35,
                early_stop=True, stop_on_pass=False, until=6, patience=1, no_emerge=True)
    return argparse.Namespace(**{**base, **kw})


def ri(root, k):
    return json.loads((root / f"harvest/round-{k}.json").read_text(encoding="utf-8"))


H = new_run("runD", rows_d)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    harvest.cmd_run(H)
rootD = RUN["root"]
d1, d2, d3 = (ri(rootD, k) for k in (1, 2, 3))
check("run D: raw yield/min flat (20 new Yüksek clusters / 10 min every round)",
      [x["ypm"] for x in (d1, d2, d3)] == [2.0, 2.0, 2.0], [x["ypm"] for x in (d1, d2, d3)])
check("run D: round 3's 20 clusters are paraphrases of round 1's -> novelty 0/20, ypm_adj 0.0; rounds 1-2 all novel",
      d3["novelty"]["new"] == 20 and d3["novelty"]["novel"] == 0 and d3["ypm_adj"] == 0.0
      and d1["novelty"]["share"] == 1.0 and d2["novelty"]["share"] == 1.0 and d2["ypm_adj"] == 2.0,
      (d1.get("novelty"), d3.get("novelty"), d3.get("ypm_adj")))
check("run D: the weak rule judges the adjusted yield/min -> round 3 stops (raw 2.0 would not have)",
      d3.get("stop") == "yield/min 0.0 < 0.35 x best round (2.0)" and not (rootD / "harvest/round-4.json").exists(),
      d3.get("stop"))
check("run D: stop_rule peak-v2-novel in every round json", all(x["stop_rule"] == "peak-v2-novel" for x in (d1, d2, d3)))
log = (rootD / "harvest/ROUNDS.md").read_text(encoding="utf-8")
check("run D: ROUNDS.md decision notes carry 'yenilik %x' (100 for round 2, 0 + duplicate yield for round 3)",
      "> tur 2: devam: verim/dk 2.0 = 1.00 x en iyi tur; yenilik %100 (tekrar verimi -0)" in log
      and "yenilik %0 (tekrar verimi -20)" in log, log[-800:])
check("run D: per-round console line 'novelty: m/n new clusters novel (x%), paraphrase-duplicate yield -y'",
      "novelty: 0/20 new clusters novel (0%), paraphrase-duplicate yield -20" in out.getvalue(), out.getvalue()[-1500:])

# an old run (rounds without ypm_adj, stop_rule peak-v1) resumed: cmd_run backfills once at start
saved = {k: ri(rootD, k) for k in (1, 2, 3)}
for k in (1, 2, 3):
    x = {kk: v for kk, v in saved[k].items() if kk not in ("novelty", "ypm_adj", "new_yield_adj", "sub_new_adj")}
    x["stop_rule"] = "peak-v1"
    write_round(rootD, k, **x)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    harvest.cmd_run(argparse.Namespace(**vars(H)))
check("resume of an old run: cmd_run logs and backfills ypm_adj / novelty into every finished round (same values)",
      "backfilling the finished rounds" in out.getvalue()
      and all(ri(rootD, k)["ypm_adj"] == saved[k]["ypm_adj"] and ri(rootD, k)["novelty"] == saved[k]["novelty"]
              for k in (1, 2, 3)), out.getvalue()[-800:])
out = io.StringIO()
with contextlib.redirect_stdout(out):
    harvest.cmd_run(argparse.Namespace(**vars(H)))
check("resume again: history already adjusted -> no second backfill", "backfill" not in out.getvalue(), out.getvalue())

# E: a novelty failure never fails the round: the raw yields stand (no ypm_adj), the weak rule uses ypm
REAL_NOV = harvest.novelty
harvest.novelty = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model missing"))
H = new_run("runE", rows_d, until=3)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    harvest.cmd_run(H)
e3 = ri(RUN["root"], 3)
check("run E: novelty raising -> rounds finish without novelty fields, raw yield/min judged (round 3 not weak)",
      "ypm_adj" not in e3 and "novelty" not in e3 and e3["weak_streak"] == 0 and "stop" not in e3, e3.get("stop"))
harvest.novelty = REAL_NOV
harvest.time = time
harvest.NOVEL_SIM = OLD_SIM

finish(RESULTS, SCRATCH)
