"""Offline checks for the v3.5 checkpoints inside long phases: corpus.Checkpoint (items not yet started are skipped
once <root>/STOP exists, items in flight finish, check() consumes STOP and raises KeyboardInterrupt), and its use in
the report phases (outline, sections analyze / write, synthesis report parts, verify_fix): a STOP mid-phase pauses
before the step that needs every item (merge, summaries, intros, notes) and a resumed phase finishes from the cache.
The sweep's own checkpoints (between batches, before each pool-round stage) are in suite_v35_sweep.py.
No LLM, no network: the per-item work functions are faked; concurrency 1 makes the order deterministic.
usage: python tests/suite_v35_checkpoint.py [scratch_dir]"""
import concurrent.futures as cf
import json
import sys

from _suite import scratch  # noqa: E402  (sets hermetic LDR_* env first)

SCRATCH = scratch("v35-checkpoint-")

from local_deep_research import corpus  # noqa: E402
from local_deep_research import outline  # noqa: E402
from local_deep_research import sections  # noqa: E402
from local_deep_research import synthesis  # noqa: E402
from local_deep_research import verify_fix  # noqa: E402

RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


def new_root(name):
    root = SCRATCH / name / "research"
    for d in ("claims", "harvest", "analysis", "report"):
        (root / d).mkdir(parents=True, exist_ok=True)
    plan = {"subquestions": [{"id": 1, "title": "One", "text": "first"}, {"id": 2, "title": "Two", "text": "second"}]}
    (root / "harvest/plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return root


def stop_after(root, n, calls, out=None):
    """a fake work item: records its args, writes STOP after its own n-th call (once; a resumed phase runs on)"""
    seen = [0]

    def f(*a, **k):
        calls.append(a)
        seen[0] += 1
        if seen[0] == n:
            (root / "STOP").write_text("", encoding="utf-8")
        return out(*a) if callable(out) else out
    return f


def paused(fn):
    try:
        fn()
    except KeyboardInterrupt:
        return True
    return False


# ---------- 1. corpus.Checkpoint ----------
root = new_root("unit")
cp = corpus.Checkpoint(root, "unit")
calls = []
work = cp.wrap(stop_after(root, 2, calls, lambda x: x * 10))
with cf.ThreadPoolExecutor(1) as ex:
    futs = [ex.submit(work, i) for i in range(5)]
    got = [cp.result(f) for f in futs]
check("Checkpoint: items after the STOP are skipped (None), the ones before keep their results",
      got == [0, 10, None, None, None] and len(calls) == 2 and cp.skipped == 3, (got, calls, cp.skipped))
check("Checkpoint.check: something skipped -> STOP consumed, KeyboardInterrupt",
      paused(cp.check) and not (root / "STOP").exists())
cp2 = corpus.Checkpoint(root, "unit")
with cf.ThreadPoolExecutor(2) as ex:
    got = [cp2.result(f) for f in [ex.submit(cp2.wrap(lambda x: x + 1), i) for i in range(4)]]
check("Checkpoint: no STOP -> every item runs, check() passes", got == [1, 2, 3, 4] and not paused(cp2.check), got)
(root / "STOP").write_text("", encoding="utf-8")
check("Checkpoint.check: STOP that appeared after the last item still pauses at the step boundary",
      paused(corpus.Checkpoint(root, "unit").check) and not (root / "STOP").exists())
cp3 = corpus.Checkpoint(root, "unit")
with cf.ThreadPoolExecutor(1) as ex:
    fu = ex.submit(cp3.wrap(lambda: (_ for _ in ()).throw(ValueError("boom"))))
    try:
        cp3.result(fu)
        raised = False
    except ValueError:
        raised = True
check("Checkpoint.result: a real error is not swallowed", raised)

# ---------- 2. sections analyze ----------
root = new_root("analyze")
OUTL = {1: {"subtopics": [{"key": "S1.1", "title": "a"}, {"key": "S1.2", "title": "b"}]},
        2: {"subtopics": [{"key": "S2.1", "title": "c"}]}}
SUBS = {1: {"id": 1, "title": "One", "text": "first"}, 2: {"id": 2, "title": "Two", "text": "second"}}
sections.context = lambda r: ("en", SUBS, OUTL, "brief")
sections.load_packs = lambda r: {}
sections.ask = lambda p, mt, kind, tries=3: "## Özet\n\n- summary"
calls = []
sections.analyze_topic = stop_after(root, 1, calls, lambda r, n, sub, s, *a: f"### {s['key']} block\n")
check("analyze: STOP after the first sub-topic -> paused before the summaries, no analysis/s<n>.md, STOP consumed",
      paused(lambda: sections.cmd_analyze(root, 1)) and len(calls) == 1
      and not list((root / "analysis").glob("s*.md")) and not (root / "STOP").exists(), calls)
calls.clear()
sections.cmd_analyze(root, 1)
check("analyze resumed: every sub-topic, both analysis files",
      len(calls) == 3 and (root / "analysis/s1.md").exists() and (root / "analysis/s2.md").exists(), calls)

# ---------- 3. sections write ----------
root = new_root("write")
(root / "STYLE.md").write_text("style", encoding="utf-8")
calls = []
sections.write_topic = stop_after(root, 1, calls, lambda r, n, mm, sub, s, *a: f"{n:02d}-{mm:02d}-x.md")
check("write: STOP after the first section -> paused before the intros (no bare-title intro written)",
      paused(lambda: sections.cmd_write(root, 1)) and len(calls) == 1
      and not list((root / "report").glob("*-00-*.md")), (calls, list((root / "report").iterdir())))
calls.clear()
rc = sections.cmd_write(root, 1)
intros = sorted(f.name for f in (root / "report").glob("*-00-*.md"))
check("write resumed: every section, the intros from the model (not bare titles)",
      rc == 0 and len(calls) == 3 and intros == ["01-00-one.md", "02-00-two.md"]
      and "summary" in (root / "report/01-00-one.md").read_text(encoding="utf-8"), (rc, calls, intros))

# ---------- 4. synthesis report parts ----------
root = new_root("parts")
TODO = [("counter", {"id": 1}), ("counter", {"id": 2}), ("summary", None)]
synthesis.parts_todo = lambda r: list(TODO)
synthesis.labels = lambda r: ({}, "en")
calls = []
synthesis.counter = stop_after(root, 1, calls, lambda r, s, L, lng: f"{s['id']:02d}-karsi-gorusler.md")
synthesis.summary = lambda r, L, lng: calls.append("summary") or "00-ozet.md"
check("report parts: STOP after the first part -> paused, the rest not started",
      paused(lambda: synthesis.report_parts(root, 1)) and calls == [(root, {"id": 1}, {}, "en")], calls)
calls.clear()
check("report parts resumed: all parts, 0 failed", synthesis.report_parts(root, 1) == 0 and len(calls) == 3, calls)

# ---------- 5. outline ----------
root = new_root("outline")
ROWS = {n: [{"id": f"S{n}-{k:04d}", "lv": "Yüksek", "src": 2, "claim": f"c{k}", "line": "x"} for k in (1, 2)]
        for n in (1, 2)}
outline.load_packs = lambda r: ROWS
outline.run_lang = lambda r: "en"
outline.chunks = lambda rows: [[r] for r in rows]
outline.render = lambda sub, rows, out: "outline\n"
calls = []
outline.themes_for_chunk = stop_after(root, 1, calls, lambda t, c, lang, cache: [{"title": "t", "ids": [c[0]["id"]]}])
merges = []


def fake_merge(sub, rows, themes, lang):
    merges.append(sub["id"])
    return {"subtopics": [{"key": "k", "ids": [r["id"] for r in rows]}], "appendix_only": []}


outline.merge = fake_merge
check("outline: STOP after the first theme call -> paused before any merge (no outline json from partial themes)",
      paused(lambda: outline.main([str(root), "--concurrency", "1"])) and len(calls) == 1 and merges == []
      and not list((root / "analysis").glob("s*-outline.json")), (calls, merges))
calls.clear()
outline.themes_for_chunk = stop_after(root, 99, calls, lambda t, c, lang, cache: [{"title": "t", "ids": [c[0]["id"]]}])
outline.merge = stop_after(root, 1, merges, lambda *a: fake_merge(*a))
check("outline: STOP during the merges -> the merge in flight is written, the rest paused",
      paused(lambda: outline.main([str(root), "--concurrency", "1"])) and len(calls) == 4
      and len(list((root / "analysis").glob("s*-outline.json"))) == 1, (calls, merges))
outline.merge = fake_merge
outline.main([str(root), "--concurrency", "1"])
check("outline resumed: both outlines", len(list((root / "analysis").glob("s*-outline.json"))) == 2)

# ---------- 6. verify_fix ----------
root = new_root("verify")
for n in (1, 2):
    (root / f"report/{n:02d}-01-x.md").write_text("text", encoding="utf-8")
verify_fix.load_packs = lambda r: {}
verify_fix.run_lang = lambda r: "en"
verify_fix.cr.load_claims = lambda r: {}
notes = []
verify_fix.write_verify = lambda r, nn, part: notes.append(nn) or ""
verify_fix.write_dogrulama = lambda r, parts, lang: notes.append("95")
calls = []
verify_fix.check_file = stop_after(root, 1, calls, {"flags": [], "applied": [], "rejected": [], "answered": True})
check("verify_fix: STOP after the first file -> paused before the notes (no 95-dogrulama)",
      paused(lambda: verify_fix.main([str(root), "--concurrency", "1"])) and len(calls) == 1 and notes == [], notes)
calls.clear()
check("verify_fix resumed: every file, the notes written",
      verify_fix.main([str(root), "--concurrency", "1"]) == 0 and len(calls) == 2 and notes == [1, 2, "95"],
      (calls, notes))

print(f"\n{RESULTS[0]} passed, {RESULTS[1]} failed  (scratch: {SCRATCH})")
sys.exit(1 if RESULTS[1] else 0)
