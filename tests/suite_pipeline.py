"""Offline checks for the code-driven pipeline: llm.py (retries, cut answers, JSON), notes.py and synthesis.py output
format with a fake model (marker headings, URL whitelist, gap parsing, bounded local-file reads), pipeline.py
(phase skipping / resume, optional vs required failures, --until, the run lock, call_main, HTML fallback), cli.py
(argument parsing, profile flags, topic creation, doctor's message for an unreachable server), telemetry without a
metrics server / GPU and the opencode agent-mode config. No LLM, no network (unreachable URLs only).
usage: python tests/test_pipeline.py [scratch_dir]"""
import contextlib
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE.parent / "integrations/opencode"))
SCRATCH = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="pipeline-test-"))
shutil.rmtree(SCRATCH, ignore_errors=True); SCRATCH.mkdir(parents=True)
DEAD = "http://127.0.0.2:9"  # refused at once (127.0.0.1:9 can hang on some hosts)
os.environ.update(LDR_LLM_BASE_URL=DEAD + "/v1", LDR_SEARXNG_URL=DEAD, LDR_LLM_MODEL="fake",
                  LDR_RESEARCH_ROOT=str(SCRATCH / "projects"), LDR_CACHE_DIR=str(SCRATCH / "cache"))
os.environ.pop("LDR_CONFIG", None)

from local_deep_research import cli, config, harvest, llm, notes, pipeline, synthesis, telemetry  # noqa: E402
import build_final_config as bfc  # noqa: E402

config.set_for_tests(config.load(env={**os.environ, "LDR_LLM_CONTEXT": "32768"}))
RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


# ---------- 1. llm.py ----------
ANSWERS, SEEN = [], []


def fake_request(messages, max_tokens, kind, json_mode=False, temperature=0.3, timeout=None, meta=None, extra=None):
    SEEN.append((max_tokens, messages, extra))
    a = ANSWERS.pop(0)
    if isinstance(a, Exception):
        raise a
    return a


real_request = llm.request
llm.request = fake_request
ANSWERS[:] = [ConnectionError("down"), ("", "stop"), ("<think>hmm</think>  answer", "stop")]
out = quiet(llm.chat_text, "q", 100, "t")
check("llm.chat_text: transport error and empty answer are retried, <think> is stripped",
      out == "answer" and len(SEEN) == 3 and SEEN[0][1][0]["role"] == "system", (out, len(SEEN)))
SEEN.clear(); ANSWERS[:] = [("cut", "length"), ("whole", "stop")]
out = quiet(llm.chat_text, "q", 1000, "t")
check("llm.chat_text: a cut answer (finish_reason length) is re-asked with a larger max_tokens, never returned",
      out == "whole" and SEEN[1][0] == 1500, SEEN)
SEEN.clear(); ANSWERS[:] = [("cut", "length")] * 3 + [("whole", "stop")]
out = quiet(llm.chat_text, "q", 6000, "t", tries=3)
check("llm.chat_text: cuts do not use up the tries (live: 14 notes lost at 6000 -> 9000 -> 13500, context half free)",
      out == "whole" and [s[0] for s in SEEN] == [6000, 9000, 13500, 20250], [s[0] for s in SEEN])
LOOP = "\n".join(["- [High] a claim that keeps coming back"] * 6 + ["- another"])
SEEN.clear(); ANSWERS[:] = [(LOOP, "length"), ("whole", "stop")]
out = quiet(llm.chat_text, "q", 6000, "t", tries=3)
check("llm.chat_text: a cut answer that loops is re-asked with the loop penalty at the same budget, not grown",
      out == "whole" and [(s[0], s[2]) for s in SEEN] == [(6000, None), (6000, {"repetition_penalty": 1.05})],
      [(s[0], s[2]) for s in SEEN])
SEEN.clear(); ANSWERS[:] = [(LOOP, "length"), ConnectionError("400 unknown field"), ("whole", "stop")]
out = quiet(llm.chat_text, "q", 6000, "t", tries=3)
check("llm.chat_text: a server that rejects the penalty field gets the plain request again",
      out == "whole" and [s[2] for s in SEEN] == [None, {"repetition_penalty": 1.05}, None], [s[2] for s in SEEN])
SEEN.clear(); ANSWERS[:] = [(LOOP, "length")] * 5
try:
    quiet(llm.chat_text, "q", 6000, "t", tries=1)
    ok = False
except llm.LLMError as e:
    ok = "looped" in str(e) and [s[2] for s in SEEN] == [None] + [{"repetition_penalty": p} for p in llm.LOOP_PENALTIES]
check("llm.chat_text: loop re-asks do not use up the tries, the penalty rises (live: S9 looped 3x at 1.05), "
      "bounded by LOOP_PENALTIES", ok, [s[2] for s in SEEN])
check("llm.looping: 5 equal lines loop; distinct lines and short '-' placeholders do not",
      llm.looping(LOOP) and not llm.looping("\n".join(f"- claim {k}" for k in range(300)) + "\n-" * 8))
SEEN.clear(); ANSWERS[:] = [("x", "length")] * 3
try:
    quiet(llm.chat_text, "q" * 90000, 2000, "t")
    ok = False
except llm.LLMError as e:
    ok = "cut" in str(e) and len(SEEN) == 2 and SEEN[1][0] < 3000  # ~30k-token prompt: grows to the room left once
check("llm.chat_text: no room for a longer answer -> LLMError (no half answer)", ok, SEEN[:1] and len(SEEN))
ANSWERS[:] = [("```json\n{\"a\": 1}\n```", "stop"), ("{\"a\": 2}", "stop")]
check("llm.chat_json: fenced JSON parsed; validator rejection re-asks",
      quiet(llm.chat_json, "q", 100, "t", validate=lambda o: o if o["a"] == 2 else (_ for _ in ()).throw(ValueError()))
      == {"a": 2})
check("llm.input_chars / cap: prompts sized below llm_context, cut at a line break",
      llm.input_chars(6000) == (32768 - 6000 - 1500) * llm.CHARS_PER_TOKEN and llm.cap("abcdefghij\nkl", 12) == "abcdefghij"
      and llm.cap("abc", 7) == "abc", llm.input_chars(6000))
llm.request = real_request

# ---------- 2. notes.py with a fake model ----------
T = SCRATCH / "topic"; R = T / "research"
for d in ("claims", "harvest", "analysis", "notes", "report"):
    (R / d).mkdir(parents=True, exist_ok=True)
(T / "localdir").mkdir()
(T / "localdir/state.md").write_text("score went 0.235 -> 0.216 (relative -8.5%)\n", encoding="utf-8")
(T / "localdir/blob.bin").write_bytes(b"\0BINARY-CONTENT")
(SCRATCH / "outside.txt").write_text("SECRET-OUTSIDE", encoding="utf-8")
with contextlib.suppress(OSError):
    (T / "localdir/link.txt").symlink_to(SCRATCH / "outside.txt")
(R / "BRIEF.md").write_text("# BRIEF: test\n\n## Question\nDoes X improve Y?\n\n## Local files\n- `localdir/`\n"
                            "- notes go to `research/notes/local-project.md`\n- `missing/file.md`\n\n"
                            "## Output language\nen\n", encoding="utf-8")
(R / "harvest/plan.json").write_text(json.dumps({"subquestions": [
    {"id": 1, "title": "Effect", "text": "Does X improve Y?"}, {"id": 2, "title": "Cost", "text": "What does X cost?"}]}))
(R / "harvest/profile.json").write_text(json.dumps({"out_lang": "en", "profile": "normal", "langs": ["de"]}))
(R / "claims/subq-1.md").write_text("# Kanıt paketi S1\n\nheader text\n\n" + "\n".join(
    f"- **{'Yüksek' if k < 3 else 'Orta'}** (2 alan) claim {k} fails out of sample — \"quote {k}\" "
    f"[https://a{k}.org/x] ⟨S1-{k:04d}⟩" for k in range(1, 6)) + "\n", encoding="utf-8")
(R / "claims/CONTRADICTIONS.md").write_text(
    "# Contradiction review (automatic)\n\nintro\n\n## Real contradiction (1)\n\n- **A:** claim [https://c1.org/a]\n"
    "  - **B:** \"q\" [https://c2.org/b]\n\n## Scope difference (market, asset, sample) (1)\n\n- **A:** s "
    "[https://c3.org/s]\n", encoding="utf-8")
PROMPTS = {}
GAP = "Which study measured the effect of X on Y between 2020 and 2025?"


def fake_text(prompt, max_tokens, kind, **kw):
    PROMPTS.setdefault(kind, []).append(prompt)
    if kind == "notes-reader":
        return (f"# S1: Effect\n\n## Özet\n\n- [High] claim 1 holds - https://a1.org/x - \"quote 1\"\n- [Medium] invented"
                f" - https://evil.example/x\n\n## Çelişkiler\n\n- none found\n\n## Boşluklar\n\n- {GAP}\n")
    if kind == "notes-skeptic":
        return ("# Skeptic\n\n## Özet\n\n- S1: claim 1 - counter https://c2.org/b - verdict: weakened\n- S2: other - "
                "https://a3.org/x\n\n## Çelişkiler\n\n- c1 vs c2\n\n## Boşluklar\n\n- Which replication tested claim 1?\n")
    if kind == "notes-local":
        return "# Local project\n\n## Özet\n\n- state.md: score 0.235 -> 0.216 (relative -8.5%)\n"
    if kind == "synthesis-style":
        return "# Style sheet\n\n## Glossary\n\n- X: the method https://a1.org/x\n"
    if kind == "synthesis-analysis":
        heads = re.findall(r"(?m)^## \(\w\) .+$", prompt)
        return "# Cross-cutting analysis\n\n## Synthesis\n\nS1 https://a1.org/x and https://made-up.org/y\n\n" + \
            "\n\n".join(f"{h}\n\n- rec https://a1.org/x" for h in heads) + "\n"
    if kind == "synthesis-counter":
        n = re.search(r"Part S(\d+)", prompt).group(1)
        return f"## Counter-arguments and limitations (S{n})\n\n- objection to claim 1: out of sample it fails https://c2.org/b\n\n**Net assessment:** holds with caveats\n"
    if kind == "synthesis-summary":
        return "# Executive summary\n\nanswer https://a1.org/x\n\n## Key findings\n\n- S1 [High] https://a1.org/x\n"
    if kind == "synthesis-conflicts":
        return ("# Contradictions and reconciliation\n\n## Real contradictions\n\n- https://c1.org/a vs https://c2.org/b"
                " https://not-in-input.org\n\n## Apparent contradictions\n\n- scope https://c3.org/s\n")
    raise AssertionError(kind)


llm.chat_text = fake_text
paths = notes.brief_paths(R)
check("notes.brief_paths: listed folder found, research/ output path and missing paths ignored",
      paths == [(T / "localdir").resolve()], paths)
lf = notes.local_files(paths)
check("notes.local_files: text files only, a symlink out of the listed folder is skipped",
      [p.name for p in lf] == ["state.md"], lf)
failed = quiet(notes.run, R, 4)
s1 = (R / "notes/s1.md").read_text(encoding="utf-8")
check("notes: reader note has the three marker headings, the invented URL is removed, real URLs kept",
      failed == 0 and all(h in s1 for h in notes.HEADS) and "evil.example" not in s1 and "https://a1.org/x" in s1, s1)
rp = PROMPTS["notes-reader"][0]
check("notes: reader prompt = nonce-wrapped pack claim lines (no pack header), sub-question, rules",
      re.search(r"<<<PACK-[0-9a-f]+>>>", rp) and "claim 3 fails" in rp and "header text" not in rp
      and "Does X improve Y?" in rp and "## Boşluklar" in rp, rp[:300])
check("notes: harvest's gap parser reads the reader note's gaps", harvest.note_gaps(s1) == [GAP], harvest.note_gaps(s1))
s2 = (R / "notes/s2.md").read_text(encoding="utf-8")
check("notes: a sub-question without a pack gets a code note whose gap is the sub-question (no model call)",
      harvest.note_gaps(s2) == ["What does X cost?"] and len(PROMPTS["notes-reader"]) == 1, s2)
sk = (R / "notes/skeptic.md").read_text(encoding="utf-8")
skp = PROMPTS["notes-skeptic"][0]
check("notes: skeptic gets the notes' Özet, the REAL conflicts only and counter lines; output keeps their URLs",
      "claim 1 holds" in skp and "https://c1.org/a" in skp and "https://c3.org/s" not in skp and "claim 4 fails" in skp
      and "https://c2.org/b" in sk and "https://a3.org/x" in sk, skp[:500])
lp = (R / "notes/local-project.md").read_text(encoding="utf-8")
lpp = PROMPTS["notes-local"][0]
check("notes: local-project read the listed text file only (no binary, no outside file); missing heads re-asked, "
      "then added", "0.235 -> 0.216" in lpp and "BINARY" not in lpp and "SECRET-OUTSIDE" not in lpp
      and len(PROMPTS["notes-local"]) == 2 and all(h in lp for h in notes.HEADS), lpp[:300])
n_calls = sum(map(len, PROMPTS.values()))
quiet(notes.run, R, 4)
check("notes: complete() and a second run asks nothing (resumable)", notes.complete(R) and
      sum(map(len, PROMPTS.values())) == n_calls)

# ---------- 3. synthesis.py with a fake model ----------
(R / "analysis/s1.md").write_text("# Analysis S1\n\n## Özet\n\n- [High] (S1.1) X helps https://a1.org/x\n\n## Outline\n\n"
                                  "### S1.1 t\n- detail https://a2.org/x\n", encoding="utf-8")
(R / "claims/local-facts.md").write_text("# facts\n- L3: score 0.235 -> 0.216\n", encoding="utf-8")
quiet(synthesis.style_and_analysis, R)
an = (R / "ANALYSIS.md").read_text(encoding="utf-8")
anp = PROMPTS["synthesis-analysis"][0]
check("synthesis: ANALYSIS.md has groups (c) / (b) / (a) (b only because local notes exist), (c) first",
      anp.index("## (c) New") < anp.index("## (b) Already") < anp.index("## (a) Standard")
      and all(h in an for h in ("## (c) New and actionable", "## (b) Already in the project", "## (a) Standard practice")),
      anp[-800:])
check("synthesis: ANALYSIS.md keeps input URLs, drops invented ones; input = Özet heads, not the outline blocks",
      "https://a1.org/x" in an and "made-up.org" not in an and "detail https://a2.org" not in anp, an)
st = (R / "STYLE.md").read_text(encoding="utf-8")
check("synthesis: STYLE.md written, carries no URL", st.startswith("# Style sheet") and "http" not in st, st)
quiet(synthesis.report_parts, R, 4)
kg = (R / "report/01-karsi-gorusler.md").read_text(encoding="utf-8")
oz = (R / "report/00-ozet.md").read_text(encoding="utf-8")
cel = (R / "report/90-celiskiler.md").read_text(encoding="utf-8")
check("synthesis: counter-arguments only for analysed parts, '##' heading, skeptic URL kept",
      kg.startswith("## Counter-arguments and limitations (S1)") and "https://c2.org/b" in kg
      and not (R / "report/02-karsi-gorusler.md").exists(), kg)
check("synthesis: executive summary is a top section and points to the evidence appendix",
      oz.startswith("# Executive summary") and "APPENDIX.md" in oz, oz)
cp = PROMPTS["synthesis-conflicts"][0]
check("synthesis: contradictions section: real conflicts first in the input, URL whitelist applied",
      cp.index("Real contradiction") < cp.index("Scope difference") and "not-in-input" not in cel
      and cel.startswith("# Contradictions and reconciliation"), cel)
check("synthesis: parts_complete after the parts; a re-run asks nothing", synthesis.parts_complete(R)
      and quiet(synthesis.report_parts, R) == 0 and len(PROMPTS["synthesis-summary"]) == 1)
(R / "notes/local-project.md").unlink(); (R / "ANALYSIS.md").unlink(); (R / "claims/CONTRADICTIONS.md").unlink()
(R / "report/90-celiskiler.md").unlink()
with contextlib.suppress(Exception):
    shutil.rmtree(T / "localdir")
quiet(synthesis.style_and_analysis, R)
check("synthesis: without local notes ANALYSIS.md has no group (b); no contradictions -> no 90 section, parts done",
      "(b)" not in PROMPTS["synthesis-analysis"][1] and synthesis.parts_complete(R)
      and not (R / "report/90-celiskiler.md").exists())
(R / "harvest/profile.json").write_text(json.dumps({"out_lang": "tr"}))
check("synthesis: Turkish output -> Turkish headings", synthesis.labels(R)[0]["summary"] == "Yönetici özeti"
      and synthesis.labels(R)[1] == "Turkish")
(R / "harvest/profile.json").write_text(json.dumps({"out_lang": "en", "profile": "normal", "langs": ["de"]}))

# ---------- 4. pipeline.py ----------
P = SCRATCH / "pipe/research"; P.mkdir(parents=True); (P / "BRIEF.md").write_text("# b\n", encoding="utf-8")
RAN = []


def mk(name, optional=False, fail=False):
    def run(ctx):
        RAN.append(name)
        if fail:
            raise RuntimeError(f"{name} broke")
        (ctx.root / f"{name}.out").write_text("x", encoding="utf-8")
    return pipeline.Phase(name, run, lambda root: (root / f"{name}.out").exists(), optional)


plist = [mk("a"), mk("b", optional=True, fail=True), mk("c"), mk("d")]
(P / "a.out").write_text("x", encoding="utf-8")
rc = quiet(pipeline.run, P.parent, plist=plist, concurrency=2, sample=False)
logt = (P / ".run.log").read_text(encoding="utf-8")
check("pipeline: done phase skipped, optional failure logged and the run goes on, .final_done at the end",
      rc == 0 and RAN == ["b", "c", "d"] and "phase a: skip" in logt and "phase b: optional, failed" in logt
      and (P / ".final_done").exists(), (rc, RAN, logt))
RAN.clear(); (P / ".final_done").unlink(); (P / "c.out").unlink(); (P / "d.out").unlink()
plist[2] = mk("c", fail=True)
rc = quiet(pipeline.run, P, plist=plist, concurrency=2, sample=False)
check("pipeline: a required failure stops the run (exit 1), later phases do not run, no .final_done",
      rc == 1 and RAN == ["b", "c"] and not (P / ".final_done").exists() and "phase c: FAILED" in
      (P / ".run.log").read_text(encoding="utf-8"), (rc, RAN))
RAN.clear(); plist[2] = mk("c")
rc = quiet(pipeline.run, P, plist=plist, until="c", concurrency=2, sample=False)
check("pipeline: resume retries from the failed phase; --until stops after it", rc == 0 and RAN == ["b", "c"]
      and not (P / "d.out").exists(), RAN)
try:
    quiet(pipeline.run, P, plist=plist, until="nope", sample=False)
    ok = False
except ValueError:
    ok = True
check("pipeline: unknown --until phase is an error", ok)
with pipeline.run_lock(P):
    try:
        quiet(pipeline.run, P, plist=plist, concurrency=2, sample=False)
        ok = False
    except RuntimeError as e:
        ok = "another" in str(e)
check("pipeline: a second run on the same topic is refused while the lock is held", ok)
check("pipeline: research_dir accepts the topic or the research folder",
      pipeline.research_dir(P.parent) == P.resolve() and pipeline.research_dir(P) == P.resolve())
names = [p.name for p in pipeline.phases()]
check("pipeline: phase order (harvest before notes, gap after notes, write after style, verify before outputs)",
      names.index("harvest") < names.index("notes") < names.index("gap") < names.index("outline")
      < names.index("style") < names.index("write") < names.index("parts") < names.index("assemble")
      < names.index("verify") < names.index("assemble-2") < names.index("pdf"), names)
A = SCRATCH / "asm/research"; (A / "report").mkdir(parents=True)
(A / "report/01-a.md").write_text("# 1. A\n", encoding="utf-8"); (A / "report/02-b.md").write_text("# 1. B\n", encoding="utf-8")
rc_asm = quiet(pipeline.call_main, "assemble_report", [A])
rc_num = quiet(pipeline.call_main, "numeric_findings", [A])
check("pipeline.call_main: main(argv) and main()-with-sys.argv modules; sys.exit('msg') -> exit 1",
      rc_asm == 0 and "# 2. B" in (A / "REPORT.md").read_text(encoding="utf-8") and rc_num == 1
      and sys.argv[0] != "numeric_findings", (rc_asm, rc_num))
check("pipeline: assemble done-test follows the section files' mtimes",
      pipeline._assemble_done(A) and (os.utime(A / "report/02-b.md", (9e9, 9e9)) or not pipeline._assemble_done(A)))
(A / "REPORT.md").write_text("# T\n\n<script>alert(1)</script>\n\n> quoted\n", encoding="utf-8")
h = pipeline.html_fallback(A / "REPORT.md").read_text(encoding="utf-8")
check("pipeline.html_fallback: raw HTML from the report is escaped, blockquotes kept",
      "<script>" not in h and "&lt;script&gt;" in h and ("<blockquote>" in h or "<pre>" in h), h[-300:])
st = pipeline.status_text(P)
check("pipeline.status_text: phases with [x] / [ ], next phase, log tail", "[ ] profile" in st and "next phase: profile"
      in st and "last log lines" in st, st[:300])

# ---------- 5. cli.py ----------
a = cli.parser().parse_args(["run", "does x improve y", "--profile", "deep", "--out-lang", "tr", "--max-rounds", "3",
                             "--until", "notes"])
check("cli: run arguments parsed", (a.target, a.profile, a.out_lang, a.max_rounds, a.until, a.func) ==
      ("does x improve y", "deep", "tr", 3, "notes", cli.cmd_run))
try:
    with contextlib.redirect_stderr(io.StringIO()):
        cli.parser().parse_args(["run", "x", "--profile", "huge"])
    ok = False
except SystemExit:
    ok = True
check("cli: unknown profile rejected by the parser", ok)
check("cli.profile_args: given flags + the existing profile's choices kept (a resumed run keeps its languages)",
      cli.profile_args(R, a) == ["--profile", "deep", "--langs", "de", "--out-lang", "tr", "--rounds", "3"],
      cli.profile_args(R, a))
b = cli.parser().parse_args(["run", str(T)])
check("cli.profile_args: no flags and a profile on disk -> [] (keep it); BRIEF's output language read",
      cli.profile_args(R, b) == [] and cli.brief_lang(R) == "en")
check("cli.slugify: ASCII folder names from any question", cli.slugify("Çok dilli İşaret: ışık / Ölçüm?") ==
      "cok-dilli-isaret-isik-olcum" and cli.slugify("???") == "topic")
tp = cli.create_topic("How do tides affect coastal erosion rates?", draft=False)
bt = (tp / "research/BRIEF.md").read_text(encoding="utf-8")
check("cli.create_topic: <projects>/<slug>/research/BRIEF.md with the question and the optional sections",
      tp.parent == (SCRATCH / "projects").resolve() and "## Local files" in bt and "## Output language" in bt
      and "\nHow do tides affect coastal erosion rates?\n" in bt, tp)
same = cli.create_topic("How do tides affect coastal erosion rates?", draft=False)
try:
    cli.create_topic("Another question entirely?", name="how-do-tides-affect-coastal-erosion-rates", draft=False)
    ok = False
except cli.UserError:
    ok = True
check("cli.create_topic: the same question reuses its folder, another question under that name is refused",
      same == tp and ok)
check("cli.find_topic: by folder or by name under the projects dir", cli.find_topic(str(tp)) == tp
      and cli.find_topic(tp.name) == tp and cli.find_topic("no-such-topic") is None)
s = config.load()
c = cli.check_llm(s)
check("cli doctor: unreachable LLM -> FAIL with the URL tried and how to configure it",
      not c[0][1] and DEAD + "/v1/models" in c[0][2] and "LDR_LLM_BASE_URL" in c[0][3] and "ldr.toml" in c[0][3], c)
c = cli.check_searx(s)
check("cli doctor: unreachable SearXNG -> FAIL with the URL and the setting", not c[0][1] and DEAD in c[0][2]
      and "LDR_SEARXNG_URL" in c[0][3], c)
err = io.StringIO()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
    rc = cli.main(["run", str(tp)])
check("cli run: unreachable servers stop before any phase, exit 2, nothing written",
      rc == 2 and "fix the problems" in err.getvalue() and not (tp / "research/.run.log").exists(), err.getvalue())
with contextlib.redirect_stdout(io.StringIO()) as o:
    rc = cli.main(["doctor", "--skip-embed"])
check("cli doctor: exit 1 and every failure printed with its fix", rc == 1 and o.getvalue().count("fix:") >= 2,
      o.getvalue())

# ---------- 6. telemetry without a metrics server / GPU ----------
(R / "dataset/llm").mkdir(parents=True, exist_ok=True)
(R / "dataset/llm/notes-reader.jsonl").write_text("\n".join(json.dumps(
    {"usage": {"prompt_tokens": 100, "completion_tokens": 10}, "latency_s": 2.0, "finish_reason": f}) for f in
    ("stop", "length")) + "\n", encoding="utf-8")
quiet(telemetry.sample_once, R, "notes")
quiet(telemetry.report, R, True)
tm = (R / "TELEMETRY.md").read_text(encoding="utf-8")
check("telemetry: a sample without /metrics or GPUs, pipeline phase names, model calls per kind from the LLM log",
      "| notes-reader | 2 | 200 | 20 | 2.0 s | 1 |" in tm and telemetry.phase_key("notes") == "notes"
      and telemetry.phase_key("harvest r2:extract; agents researcherx7") == "harvest r2 extract + researcher"
      and "## Agent token use" not in tm, tm[-600:])

# ---------- 7. opencode agent-mode config ----------
O = SCRATCH / "oc"
ag = bfc.build(O, SCRATCH / "projects", "/venv/bin/python", model="m", base_url=DEAD + "/v1")
oc = json.loads((O / "opencode/opencode.json").read_text(encoding="utf-8"))
sk_ = (O / "opencode/skills/deep-research/SKILL.md").read_text(encoding="utf-8")
rd = ag["researcher"]["permission"]["read"]
allf = "".join(p.read_text(encoding="utf-8") for p in O.rglob("*") if p.is_file())
check("opencode config: generic (paths / server from arguments), agents run `ldr` commands only, packs denied",
      oc["provider"]["ldr"]["options"]["baseURL"] == DEAD + "/v1" and "{LDR}" not in sk_
      and "/venv/bin/python -m local_deep_research run" in sk_ and rd.get("research/claims/subq-2-p3.md") is None
      and rd.get(f"{(SCRATCH / 'projects').resolve()}/**/claims/subq-*.md") == "deny"
      and ag["orchestrator"]["permission"]["bash"]["*"] == "deny", rd)

print(f"\n{RESULTS[0]} passed, {RESULTS[1]} failed  (scratch: {SCRATCH})")
sys.exit(1 if RESULTS[1] else 0)
