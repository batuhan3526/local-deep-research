"""Offline checks for cut answers (finish_reason "length"): outline / sections / verify_fix retry with a doubled
budget bounded by the server's context, give up when there is no room, and a fallback outline theme is not cached.
Live run 2026-10-20: Turkish theme summaries in indented JSON ran past the fixed budget and three equal retries all
cut, so whole chunks became "Unsorted evidence" - and that fallback was cached for every later run.
usage: python tests/suite_retries.py"""
import contextlib
import io
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from local_deep_research import config, corpus, outline, sections, verify_fix  # noqa: E402

RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


config.set_for_tests(config.Settings(llm_model="m", llm_context=32768))
check("fit_budget: capped by the context left after the prompt, floor 256",
      corpus.fit_budget(16000, "x" * 60000) == 32768 - 20000 - 512 and corpus.fit_budget(9000, "x" * 99000) == 256)
check("grow_budget: doubles, stops at MAX_OUT / the context, None when no room",
      corpus.grow_budget(3000, "p") == 6000 and corpus.grow_budget(12000, "p") == corpus.MAX_OUT
      and corpus.grow_budget(12256, "x" * 60000) is None)
check("check_cut: finish_reason length raises AnswerCut, stop does not",
      isinstance(corpus.AnswerCut("x"), ValueError)
      and (lambda: (corpus.check_cut({"choices": [{"finish_reason": "stop"}]}), True))()[1])
try:
    corpus.check_cut({"choices": [{"finish_reason": "length"}]}); cut = False
except corpus.AnswerCut:
    cut = True
check("check_cut raises on a cut answer", cut)

for mod, call in ((outline, lambda: outline.ask("p", 1500, "k")), (sections, lambda: sections.ask("p", 1500, "k")),
                  (verify_fix, lambda: verify_fix.ask("p", 1500))):
    seen = []

    def fake(prompt, max_tokens, *a, _seen=seen, _mod=mod):
        _seen.append(max_tokens)
        if max_tokens < 5000:
            raise corpus.AnswerCut("cut")
        return {"ok": 1} if _mod is not sections else "text"
    mod.llm = fake
    with contextlib.redirect_stderr(io.StringIO()):
        res = call()
    check(f"{mod.__name__.split('.')[-1]}.ask: a cut answer is retried with a doubled budget and then succeeds",
          seen == [1500, 3000, 6000] and res, seen)

seen = []


def always_cut(prompt, max_tokens, *a):
    seen.append(max_tokens)
    raise corpus.AnswerCut("cut")


outline.llm = always_cut
with contextlib.redirect_stderr(io.StringIO()):
    res = outline.ask("x" * 60000, 9000, "k", tries=5)
check("ask: no room to grow -> gives up at once instead of re-sending the same budget",
      res is None and seen == [9000, 12256], seen)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="retry-test-"))
rows = [{"id": f"S1-{k:04d}", "lv": "Orta", "src": 1, "claim": f"claim {k} about the siege", "line": ""} for k in range(1, 6)]
cache = tmp / "c1.json"
outline.llm = lambda *a: (_ for _ in ()).throw(ValueError("down"))
with contextlib.redirect_stderr(io.StringIO()):
    th = outline.themes_for_chunk("why did it fail", rows, "English", cache)
check("outline: model down -> one 'Unsorted evidence' theme holds every id, and it is NOT cached",
      len(th) == 1 and len(th[0]["ids"]) == 5 and not cache.exists(), (th, cache.exists()))
outline.llm = lambda *a: {"themes": [{"title": "Siege", "summary": "s", "ids": [r["id"] for r in rows]}]}
th = outline.themes_for_chunk("why did it fail", rows, "English", cache)
check("outline: the next run asks again and caches a real answer",
      th[0]["title"] == "Siege" and json.loads(cache.read_text(encoding="utf-8"))["themes"][0]["title"] == "Siege")

# harvest.llm_json (plan / gap / translate calls): live run 2026-10-21, a thinking model spent the whole 6000-token
# plan budget on reasoning, the answer was empty and the run stopped at phase plan
from local_deep_research import harvest  # noqa: E402


class FakeResp:
    def __init__(self, finish, content):
        self._j = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


sent = []


def fake_post(url, json=None, **k):
    sent.append(json["max_tokens"])
    return FakeResp("length", "") if json["max_tokens"] < 12000 else FakeResp("stop", '{"ok": 1}')
harvest.httpx.post = fake_post
with contextlib.redirect_stderr(io.StringIO()):
    res = harvest.llm_json("plan p")
check("harvest.llm_json: an answer cut at max_tokens is retried with a doubled budget and then parsed",
      res == {"ok": 1} and sent == [6000, 12000], sent)
sent.clear()
try:
    with contextlib.redirect_stderr(io.StringIO()):
        harvest.llm_json("plan p", 1000)
    raised = False
except corpus.AnswerCut:
    raised = True
check("harvest.llm_json: still cut after the last try -> AnswerCut (callers keep their error handling)",
      raised and sent == [1000, 2000, 4000], sent)

print(f"\n{RESULTS[0]} passed, {RESULTS[1]} failed")
sys.exit(1 if RESULTS[1] else 0)
