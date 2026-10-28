"""Offline checks for v3.5 yield rule 2 and own-history sub-question weights: srcclass.practitioner (repo / docs /
help-centre / registry URLs, never farms, issues or user pages), row_yield rule 1 vs 2, yield_rebases (bare v3.1 dict
and list), rebase_factor (global and per sub-question, chained rebases), rule_at, ensure_yield_rule (legacy run ->
one rebase with factors and sub_base, idempotent; new run -> just the rule), sub_hist on a rebased history,
sub_weights "own" vs "share".
No LLM, no network, no model: embeddings are deterministic vectors.
usage: python tests/suite_v35_yield.py [scratch_dir]"""
import csv
import hashlib
import json
import re

import numpy as np

from _suite import finish, scratch  # noqa: E402  (sets hermetic LDR_* env first)

SCRATCH = scratch("v35-test-")

from local_deep_research import corpus  # noqa: E402
from local_deep_research import harvest  # noqa: E402
from local_deep_research import rt_embed  # noqa: E402
from local_deep_research.srcclass import practitioner  # noqa: E402

RESULTS = [0, 0]
print("harvest module:", harvest.__file__)


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


def fake_embed(texts, model=None, device=None):  # "T<k>" -> axis k; anything else hashed onto 64 spare axes
    out = []
    for t in texts:
        v = np.zeros(96)
        ts = re.findall(r"\bT(\d+)\b", t)
        for k in ts:
            v[int(k)] += 1.0
        if not ts:
            for w in re.findall(r"\w+", t.lower()):
                v[32 + int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 64] += 1
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out)


rt_embed.embed = corpus.embed = fake_embed


def write_round(root, k, **info):
    (root / "harvest").mkdir(parents=True, exist_ok=True)
    harvest.round_file(root, k).write_text(json.dumps(info), encoding="utf-8")


S = lambda i, **kw: {"id": i, "title": f"t{i}", "text": f"T{i}", **kw}  # noqa: E731

# ---------- 1. practitioner sources ----------
YES = ["https://github.com/smicallef/spiderfoot", "https://github.com/laramies/theHarvester/blob/master/README.md",
       "https://gitlab.com/owner/tool/-/tree/main".replace("/-", ""), "https://github.com/o/r/wiki/Usage",
       "https://help.shodan.io/the-basics/search-query-fundamentals", "https://docs.censys.com/docs/overview",
       "https://support.recordedfuture.com/hc/en-us", "https://example-vendor.com/docs/api/search",
       "https://vendor.io/en/help/getting-started", "https://pypi.org/project/holehe/", "https://tool.readthedocs.io/x"]
NO = ["https://github.com/o/r/issues/12", "https://github.com/someuser", "https://github.com/o/r/pulls",
      "https://vendor.com/blog/top-10-osint-tools", "https://www.bellingcat.com/resources/2024/x",
      "https://arxiv.org/abs/2401.00001", "https://mljourney.com/docs/guide", "not a url", "", None]
check("practitioner: repo / docs / help / support / registry pages", all(practitioner(u) for u in YES),
      [u for u in YES if not practitioner(u)])
check("practitioner: not issues, pulls, user pages, blogs, papers, farm hosts, junk", not any(practitioner(u) for u in NO),
      [u for u in NO if practitioner(u)])

# ---------- 2. row_yield ----------
R = lambda conf, urls, n=1: {"confidence": conf, "urls": " ".join(urls), "sources": str(n)}  # noqa: E731
GH, ARX, BLOG, BLOG2 = YES[0], "https://arxiv.org/abs/2401.00001", NO[3], "https://other.net/post"
cases = [  # (row, rule 1, rule 2)
    (R("Yüksek", [BLOG, BLOG2], 3), 1.0, 1.0),
    (R("Orta", [ARX]), 0.25, 0.25),
    (R("Orta", [GH]), 0.0, 0.25),
    (R("Orta", [BLOG]), 0.0, 0.0),
    (R("Orta", [BLOG, BLOG2], 2), 0.0, 0.25),
    (R("Orta", [GH, BLOG], 2), 0.0, 0.5),
    (R("Orta", [ARX, BLOG], 2), 0.25, 0.5),
    (R("Düşük", [GH, ARX], 2), 0.0, 0.0),
    ({"confidence": "Orta", "urls": GH, "sources": "x"}, 0.0, 0.25),
]
got = [(harvest.row_yield(r, 1), harvest.row_yield(r)) for r, _, _ in cases]
check("row_yield: rule 1 unchanged, rule 2 = practitioner sources + corroboration without an anchor",
      got == [(a, b) for _, a, b in cases], list(zip(got, [(a, b) for _, a, b in cases])))
check("row_yield: default rule is YIELD_RULE = 2", harvest.YIELD_RULE == 2 and harvest.row_yield(cases[2][0]) == 0.25)

# ---------- 3. yield_rebases / rebase_factor / rule_at ----------
R1 = SCRATCH / "rb/research"; (R1 / "harvest").mkdir(parents=True)
check("yield_rebases: none -> [], rule_at -> YIELD_RULE, factor 1",
      harvest.yield_rebases(R1) == [] and harvest.rule_at(R1, 5) == 2 and harvest.rebase_factor(R1, 1, 9) == 1.0)
(R1 / "harvest/yield_rebase.json").write_text(json.dumps({"factor": 0.5, "from_round": 3}), encoding="utf-8")
check("yield_rebases: a bare v3.1 dict reads as one rebase; yield_rebase() = it",
      harvest.yield_rebases(R1) == [{"factor": 0.5, "from_round": 3}] and harvest.yield_rebase(R1)["factor"] == 0.5)
check("rule_at: a rebase without 'rule' (v3.1 label change) does not change the yield rule", harvest.rule_at(R1, 1) == 2)
(R1 / "harvest/yield_rebase.json").write_text(json.dumps({"rebases": [
    {"factor": 0.5, "from_round": 3},
    {"rule": 2, "from_rule": 1, "from_round": 6, "factor": 1.2, "sub_factor": {"1": 2.0, "2": 1.0}}]}), encoding="utf-8")
check("rebase_factor: product of the rebases with k < from_round <= rnd",
      harvest.rebase_factor(R1, 2, 9) == 0.5 * 1.2 and harvest.rebase_factor(R1, 4, 9) == 1.2
      and harvest.rebase_factor(R1, 2, 4) == 0.5 and harvest.rebase_factor(R1, 6, 9) == 1.0
      and harvest.rebase_factor(R1, 2, 2) == 1.0,
      [harvest.rebase_factor(R1, k, r) for k, r in ((2, 9), (4, 9), (2, 4), (6, 9), (2, 2))])
check("rebase_factor: per sub-question factor, falling back to the global one (v3.1 rebase has none)",
      harvest.rebase_factor(R1, 4, 9, 1) == 2.0 and harvest.rebase_factor(R1, 2, 9, 1) == 0.5 * 2.0
      and harvest.rebase_factor(R1, 4, 9, 7) == 1.2)
check("rule_at: rounds before the rule-2 rebase -> 1, from it on -> 2",
      [harvest.rule_at(R1, k) for k in (1, 5, 6, 9)] == [1, 1, 2, 2])
for k, y in ((1, 2.0), (4, 3.0), (6, 3.5)):
    write_round(R1, k, ypm=y)
check("ref_ypm: chained rebases (round 1: 2.0 x 0.6 = 1.2, round 4: 3.0 x 1.2 = 3.6 > round 6's 3.5)",
      abs(harvest.ref_ypm(R1, 7) - 3.6) < 1e-9, harvest.ref_ypm(R1, 7))

# ---------- 4. sub_hist on a rebased history ----------
R2 = SCRATCH / "sh/research"
for k, v in {1: {"1": 4, "2": 10}, 2: {"1": 4, "2": 10}, 3: {"1": 8, "2": 10}}.items():
    write_round(R2, k, sub_new=v)
(R2 / "harvest/yield_rebase.json").write_text(json.dumps({"rebases": [
    {"rule": 2, "from_rule": 1, "from_round": 3, "factor": 1.1, "sub_factor": {"1": 2.0}}]}), encoding="utf-8")
check("sub_hist: rounds before the rebase scaled by the sub-question's own factor (S1 x2, S2 global x1.1)",
      harvest.sub_hist(R2, S(1), 3) == [8.0, 8.0, 8.0] and harvest.sub_hist(R2, S(2), 3) == [11.0, 11.0, 10.0]
      and harvest.sub_hist(R2, S(1), 2) == [4.0, 4.0],
      (harvest.sub_hist(R2, S(1), 3), harvest.sub_hist(R2, S(2), 3), harvest.sub_hist(R2, S(1), 2)))

# ---------- 5. sub_weights: own history vs share ----------
R3 = SCRATCH / "sw/research"
for k, v in {1: {"1": 100, "2": 4, "3": 0}, 2: {"1": 50, "2": 4, "3": 0}}.items():
    write_round(R3, k, sub_new=v)
subs = [S(1), S(2), S(3)]
prev = harvest.round_info(R3, 2)
own = harvest.sub_weights(subs, prev, 3, R3)
share = harvest.sub_weights(subs, prev, 3, R3, mode="share")
check("sub_weights own: steady small sub (4, 4) outweighs a big one at half its best (100, 50); share is the reverse",
      own[2] > own[1] and share[1] > share[2], (own, share))
check("sub_weights own: a sub-question with no yield yet is neutral (ratio 1 = the steady one)", own[3] == own[2], own)
check("sub_weights own: mean 1", abs(sum(own.values()) / 3 - 1) < 0.01, own)
check("sub_weights: without root -> share (old callers), None before round 2 / without sub_new",
      harvest.sub_weights(subs, prev, 3) == share and harvest.sub_weights(subs, prev, 1, R3) is None
      and harvest.sub_weights(subs, {}, 3, R3) is None)
fl = [S(1, plateau=2), S(2), S(3, born=3)]
w = harvest.sub_weights(fl, prev, 3, R3)
check("sub_weights own: plateaued -> PLATEAU_W, newborn -> EMERGE_W",
      w[1] == harvest.PLATEAU_W and w[3] == harvest.EMERGE_W, w)
check("SUB_WEIGHT_MODE default is 'own'", harvest.SUB_WEIGHT_MODE == "own")

# ---------- 6. ensure_yield_rule ----------
CLAIM_COLS = ["confidence", "sources", "claims", "claim", "kind", "domains", "farm", "urls", "quote", "suspect",
              "corroborated_by", "contradicted_by", "anchored"]


def write_claims(root, rows):
    (root / "claims").mkdir(parents=True, exist_ok=True)
    with open(root / "claims/claims.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CLAIM_COLS)
        w.writeheader()
        for conf, n, claim, urls in rows:
            w.writerow({"confidence": conf, "sources": n, "claims": 1, "claim": claim, "kind": "finding",
                        "domains": "", "farm": "", "urls": " ".join(urls), "quote": "q", "anchored": 0})


ROWS = [("Yüksek", 2, "T1 a", [ARX, BLOG]), ("Orta", 1, "T1 b", [ARX]), ("Orta", 1, "T2 c", [GH]),
        ("Orta", 2, "T2 d", [GH, BLOG]), ("Orta", 1, "T2 e", [BLOG])]
R4 = SCRATCH / "ens/research"
write_claims(R4, ROWS)
for k in (1, 2):
    write_round(R4, k, ypm=1.0, sub_new={"1": 1, "2": 0})
subs4 = [S(1), S(2)]
rb = harvest.ensure_yield_rule(R4, subs4, 3)
# rule 1: S1 1 + 0.25, S2 0 -> 1.25 ; rule 2: S1 1.25, S2 0.25 + 0.5 = 0.75 -> 2.0
check("ensure_yield_rule: legacy run -> rebase rule 1 -> 2 from round 3, factor 2.0 / 1.25, sub factors, sub_base",
      rb and rb["from_rule"] == 1 and rb["rule"] == 2 and rb["from_round"] == 3 and rb["factor"] == 1.6
      and rb["sub_factor"] == {"1": 1.0} and rb["sub_base"] == {"1": 1.25, "2": 0.75}, rb)
check("ensure_yield_rule: yield_rule.json = 2, rule_at old rounds 1, new 2",
      json.loads((R4 / "harvest/yield_rule.json").read_text())["rule"] == 2
      and harvest.rule_at(R4, 2) == 1 and harvest.rule_at(R4, 3) == 2)
check("ensure_yield_rule: idempotent (second call writes nothing)",
      harvest.ensure_yield_rule(R4, subs4, 4) is None and len(harvest.yield_rebases(R4)) == 1)
check("sub_hist after the rebase: S2 had 0 under rule 1, no factor -> global 1.6 x 0 = 0",
      harvest.sub_hist(R4, S(2), 3) == [0.0, 0.0, 0.0], harvest.sub_hist(R4, S(2), 3))
R5 = SCRATCH / "new/research"
write_claims(R5, ROWS); (R5 / "harvest").mkdir()
check("ensure_yield_rule: new run (no finished rounds) -> just records the rule, no rebase",
      harvest.ensure_yield_rule(R5, subs4, 1) is None and harvest.yield_rebases(R5) == []
      and json.loads((R5 / "harvest/yield_rule.json").read_text())["rule"] == 2)
write_round(R5, 1, ypm=1.0)
check("ensure_yield_rule: a run started under rule 2 is never rebased later",
      harvest.ensure_yield_rule(R5, subs4, 2) is None and harvest.yield_rebases(R5) == [])
R6 = SCRATCH / "keep/research"
write_claims(R6, ROWS)
write_round(R6, 1, ypm=1.0)
(R6 / "harvest/yield_rebase.json").write_text(json.dumps({"factor": 0.5, "from_round": 1}), encoding="utf-8")
harvest.ensure_yield_rule(R6, subs4, 2)
check("ensure_yield_rule: an older v3.1 rebase is kept, the new one appended",
      [r.get("rule") for r in harvest.yield_rebases(R6)] == [None, 2], harvest.yield_rebases(R6))

finish(RESULTS, SCRATCH)
