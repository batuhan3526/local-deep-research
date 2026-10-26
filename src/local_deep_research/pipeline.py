"""The phase runner behind `ldr run`: the whole pipeline as plain Python, resumable from the files it writes.

Every phase has a done-test on its outputs (files, or files newer than their inputs) and is skipped when it passes,
so a crash, Ctrl-C or reboot loses at most the phase in flight; most phases cache per call inside too. Each phase
logs one line to research/.run.log and stdout. A failed required phase stops the run (exit 1, the next `ldr run`
retries it); a failed optional phase (contradiction review, numeric table, checks, appendix, PDF / HTML, XLSX /
PPTX, dataset, telemetry, the gap round) is logged and the run goes on. One run per topic at a time
(research/.run.lock, an OS file lock: released when the process dies, so a crash never leaves a stale lock).
Why code and not an agent (live 2026-10-20): a 27B model with 64k context driving the phases read too much,
compacted, skipped phases and wrote nothing; here every model call is bounded and single-purpose (llm.py).
Phase modules are called in-process (their main(argv) / functions); only md2pdf runs as a subprocess
(sys.executable -m), because WeasyPrint's native libraries may be missing or crash, and the run must survive that.

Phases (status / --until use these names):
  profile plan harvest contradictions notes local-facts gap contradictions-2 numeric outline analyze style write
  parts assemble verify assemble-2 check appendix pdf extras dataset telemetry
"""
import contextlib
import datetime
import html
import importlib
import inspect
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from typing import Callable, NamedTuple

from . import stages


class Phase(NamedTuple):
    name: str
    run: Callable      # run(ctx) -> None; raises on failure
    done: Callable     # done(root) -> bool: outputs exist (and are newer than their inputs)
    optional: bool = False
    about: str = ""


class Ctx(NamedTuple):
    root: pathlib.Path
    conc: int
    profile_args: list  # harvest profile flags to (re)write harvest/profile.json with; [] = keep the file


class Incomplete(RuntimeError):
    """a phase ran but some of its outputs could not be written (a re-run retries only those)"""


# ---------- helpers ----------
def research_dir(path):
    """<topic>/research for a topic folder, the folder itself for a research folder (has BRIEF.md / harvest/)"""
    p = pathlib.Path(path).expanduser().resolve()
    if p.name != "research" and (p / "research").is_dir():
        return p / "research"
    return p


def mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def newer(out, *inputs):
    """out exists and is at least as new as every existing input"""
    t = mtime(out)
    return t is not None and all(t >= m for m in map(mtime, inputs) if m is not None)


def touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(datetime.datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")


def log(root, msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with contextlib.suppress(OSError):
        with open(root / ".run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def call_main(name, args):
    """run local_deep_research.<name>.main with CLI args in-process -> exit code. The modules' mains differ: main()
    reading sys.argv, main(argv) with argv[0] = program name, main(argv=None) handing argv to argparse; all three
    are supported, so a module can move to the argparse form without breaking this."""
    mod = importlib.import_module(f".{name}", __package__)
    params = list(inspect.signature(mod.main).parameters.values())
    args = [str(a) for a in args]
    old = sys.argv
    try:
        if not params:
            sys.argv = [name, *args]
            rc = mod.main()
        elif params[0].default is inspect.Parameter.empty:
            rc = mod.main([name, *args])
        else:
            rc = mod.main(args)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        rc = e.code
    finally:
        sys.argv = old
    return int(rc or 0)


def must(name, args):
    rc = call_main(name, args)
    if rc:
        raise Incomplete(f"{name} exited {rc}")


def subs(root):
    from .notes import subquestions
    return subquestions(root)


def with_pack(root):
    return [s for s in subs(root) if (root / f"claims/subq-{s['id']}.md").exists()]


def analysis_started(root):
    return any(not p.name.endswith("-outline.md") for p in (root / "analysis").glob("s*.md"))


def report_files(root):
    return sorted((root / "report").glob("[0-9][0-9]-*.md"))


# ---------- phases ----------
def _profile(ctx):
    if ctx.profile_args or not (ctx.root / "harvest/profile.json").exists():
        must("harvest", ["profile", ctx.root, *ctx.profile_args])


def _plan(ctx):
    from . import harvest, notes
    p = notes.profile(ctx.root)
    harvest.plan(ctx.root, int(p.get("subs") or 6), int(p.get("per_sub") or 14))


def _harvest(ctx):
    from . import notes
    rounds = int(notes.profile(ctx.root).get("rounds") or 2)
    must("harvest", ["run", ctx.root, "--until", rounds, "--concurrency", ctx.conc])
    touch(ctx.root / "harvest/.done")


def _contradictions(ctx):
    must("corpus", ["contradictions", ctx.root, "--concurrency", ctx.conc])


def _contradictions_done(root):
    claims = root / "claims/CLAIMS.md"
    return not claims.exists() or newer(root / "claims/CONTRADICTIONS.md", claims)


def _notes(ctx):
    from . import notes
    failed = notes.run(ctx.root, ctx.conc)
    if failed:
        raise Incomplete(f"{failed} note(s) could not be written")


def _notes_done(root):
    from . import notes
    return notes.complete(root)


def _local_facts(ctx):
    must("local_facts", [ctx.root])


def _local_facts_done(root):
    src = root / "notes/local-project.md"
    return not src.exists() or newer(root / "claims/local-facts.md", src)


def _gap(ctx):
    must("harvest", ["gap", ctx.root, "--concurrency", ctx.conc])
    touch(ctx.root / "harvest/.gap_done")


def _gap_done(root):
    # analysis already started (e.g. a run resumed past it): a gap round now would come after the analysts
    return (root / "harvest/.gap_done").exists() or (root / "harvest/gap.json").exists() or analysis_started(root)


def _numeric(ctx):
    must("numeric_findings", [ctx.root])


def _outline(ctx):
    must("outline", [ctx.root, "--concurrency", ctx.conc])


def _outline_done(root):
    s = with_pack(root)
    return bool(s) and all((root / f"analysis/s{x['id']}-outline.json").exists() for x in s)


def _analyze(ctx):
    must("sections", ["analyze", ctx.root, "--concurrency", ctx.conc])


def _analyze_done(root):
    s = [x for x in subs(root) if (root / f"analysis/s{x['id']}-outline.json").exists()]
    return bool(s) and all((root / f"analysis/s{x['id']}.md").exists() for x in s)


def _style(ctx):
    from . import synthesis
    synthesis.style_and_analysis(ctx.root)


def _style_done(root):
    from . import synthesis
    return synthesis.style_complete(root)


def _write(ctx):
    must("sections", ["write", ctx.root, "--concurrency", ctx.conc])
    touch(ctx.root / "report/.sections_done")


def _parts(ctx):
    from . import synthesis
    failed = synthesis.report_parts(ctx.root, ctx.conc)
    if failed:
        raise Incomplete(f"{failed} report part(s) could not be written")


def _parts_done(root):
    from . import synthesis
    return synthesis.parts_complete(root)


def _assemble(ctx):
    must("assemble_report", [ctx.root])


def _assemble_done(root):
    return newer(root / "REPORT.md", *report_files(root)) if report_files(root) else False


def _verify(ctx):
    must("verify_fix", [ctx.root, "--concurrency", ctx.conc])


def _check(ctx):
    must("check_report", [ctx.root])


def _appendix(ctx):
    must("appendix", [ctx.root])


def html_fallback(md):
    """REPORT.html next to REPORT.md when no PDF can be built. The Markdown carries web quotes and model text, so
    raw HTML in it is escaped before conversion (a quoted <script> stays text); blockquote markers are restored."""
    text = html.escape(md.read_text(encoding="utf-8", errors="replace"), quote=False)
    text = re.sub(r"(?m)^((?:&gt;\s?)+)", lambda m: m.group(1).replace("&gt;", ">"), text)
    try:
        import markdown
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    except ImportError:
        body = f"<pre>{text}</pre>"
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Report</title><style>body{max-width:60rem;"
            "margin:2rem auto;padding:0 1rem;font-family:sans-serif;line-height:1.5}table{border-collapse:collapse}"
            "td,th{border:1px solid #999;padding:4px 6px;vertical-align:top}pre{white-space:pre-wrap}</style></head>"
            f"<body>{body}</body></html>\n")
    out = md.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    return out


def _pdf(ctx):
    md = ctx.root / "REPORT.md"
    try:
        from .corpus import module_cmd
        cmd, env = module_cmd("md2pdf", str(md))  # same import path as this process, -X utf8
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
        ok, why = r.returncode == 0, (r.stderr or "").strip().splitlines()[-1:] or [""]
    except (OSError, subprocess.TimeoutExpired) as e:
        ok, why = False, [str(e)]
    if ok and newer(ctx.root / "REPORT.pdf", md):
        return
    out = html_fallback(md)
    print(f"pdf: not built ({why[0][:200]}); wrote {out.name} instead (PDF needs: pip install "
          f"'local-deep-research[reports]' and Pango)")


def _pdf_done(root):
    md = root / "REPORT.md"
    return newer(root / "REPORT.pdf", md) or newer(root / "REPORT.html", md)


def _extras(ctx):
    must("export_extras", [ctx.root])


def _dataset(ctx):
    from . import dataset
    dataset.export(ctx.root, do_zip=True)


def _telemetry(ctx):
    from . import telemetry
    telemetry.report(ctx.root)


def phases():
    R = lambda rel: (lambda root: (root / rel).exists())  # noqa: E731
    N = lambda out, *ins: (lambda root: newer(root / out, *(root / i for i in ins)))  # noqa: E731
    return [
        Phase("profile", _profile, R("harvest/profile.json"), about="harvest/profile.json (sizes, languages)"),
        Phase("plan", _plan, R("harvest/plan.json"), about="harvest/plan.json (sub-questions + queries)"),
        Phase("harvest", _harvest, R("harvest/.done"), about="search / fetch / extract / verify rounds -> claims/"),
        Phase("contradictions", _contradictions, _contradictions_done, True, "claims/CONTRADICTIONS.md"),
        Phase("notes", _notes, _notes_done, True, "notes/s<n>.md, skeptic.md, local-project.md"),
        Phase("local-facts", _local_facts, _local_facts_done, True, "claims/local-facts.md"),
        Phase("gap", _gap, _gap_done, True, "one more harvest round from the notes' gaps"),
        Phase("contradictions-2", _contradictions, _contradictions_done, True, "review the gap round's conflicts"),
        Phase("numeric", _numeric, N("data/numeric-findings.csv", "claims/claims.csv"), True,
              "data/numeric-findings.csv"),
        Phase("outline", _outline, _outline_done, about="analysis/s<n>-outline.json"),
        Phase("analyze", _analyze, _analyze_done, about="analysis/s<n>.md"),
        Phase("style", _style, _style_done, about="STYLE.md, ANALYSIS.md"),
        Phase("write", _write, R("report/.sections_done"), about="report/NN-MM-*.md sections"),
        Phase("parts", _parts, _parts_done, about="report/NN-karsi-gorusler.md, 00-ozet.md, 90-celiskiler.md"),
        Phase("assemble", _assemble, _assemble_done, about="REPORT.md"),
        Phase("verify", _verify, R("report/95-dogrulama.md"), about="VERIFY-NN.md, report/95-dogrulama.md"),
        Phase("assemble-2", _assemble, _assemble_done, about="REPORT.md with the verification fixes"),
        Phase("check", _check, N("CHECK.md", "REPORT.md"), True, "CHECK.md"),
        Phase("appendix", _appendix, N("APPENDIX.md", "claims/claims.csv"), True, "APPENDIX.md / .pdf"),
        Phase("pdf", _pdf, _pdf_done, True, "REPORT.pdf (REPORT.html when PDF is unavailable)"),
        Phase("extras", _extras, N("REPORT.xlsx", "REPORT.md"), True, "REPORT.xlsx, REPORT.pptx"),
        Phase("dataset", _dataset, N("dataset.zip", "REPORT.md"), True, "dataset/ + dataset.zip"),
        Phase("telemetry", _telemetry, N("TELEMETRY.md", "REPORT.md"), True, "TELEMETRY.md"),
    ]


# ---------- runner ----------
@contextlib.contextmanager
def run_lock(root):
    """exclusive OS lock on research/.run.lock (fcntl / msvcrt): a second `ldr run` on the topic exits at once"""
    f = open(root / ".run.lock", "a+", encoding="utf-8")
    try:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"another `ldr run` is working on {root.parent} (lock {root / '.run.lock'})") from None
        yield
    finally:
        f.close()


def run(root, profile_args=(), until=None, plist=None, concurrency=None, sample=True):
    """run every phase whose outputs are missing -> exit code (0 = done or stopped at `until`, 1 = a required
    phase failed). KeyboardInterrupt is logged and re-raised."""
    from . import config, corpus, llmlog
    root = research_dir(root)
    if not (root / "BRIEF.md").exists():
        raise FileNotFoundError(f"no BRIEF.md in {root} (create the topic with `ldr new \"question\"`)")
    plist = phases() if plist is None else plist
    if until and until not in {p.name for p in plist}:
        raise ValueError(f"unknown phase {until!r}: {', '.join(p.name for p in plist)}")
    conc = int(concurrency or getattr(config.get(), "llm_concurrency", 16) or 16)
    ctx = Ctx(root, conc, list(profile_args))
    llmlog.set_root(root)
    current = ["start"]
    with run_lock(root):
        log(root, f"run start: {root.parent} (pid {os.getpid()})")
        stop_sampler = start_sampler(root, lambda: current[0]) if sample else (lambda: None)
        switch = stages.Switcher(lambda m: log(root, m))
        order = [p.name for p in plist]
        try:
            for p in plist:
                if p.name == "profile" and ctx.profile_args:
                    pass  # explicit size / language flags: rewrite the profile even though one exists
                elif p.done(root):
                    log(root, f"phase {p.name}: skip (done)")
                    if p.name == until:
                        break
                    continue
                if corpus.pause_requested(root):  # a graceful stop asked between phases (touch <root>/STOP)
                    log(root, f"stop requested (STOP file): paused before phase {p.name}; `ldr run` resumes here")
                    raise KeyboardInterrupt
                current[0] = p.name
                try:
                    switch.enter(stages.stage_of(p.name, order))
                    if not concurrency:  # the stage's own llm_concurrency, unless --concurrency was given
                        ctx = ctx._replace(conc=int(getattr(config.get(), "llm_concurrency", 16) or 16))
                except Exception as e:  # noqa: BLE001  a stage server that does not come up stops the run
                    log(root, f"phase {p.name}: FAILED (stage switch: {type(e).__name__}: {str(e)[:300]}); fix the "
                              f"[collect] / [report] settings or the server and run `ldr run` again to resume")
                    return 1
                t0 = time.monotonic()
                log(root, f"phase {p.name}: start")
                try:
                    p.run(ctx)
                except KeyboardInterrupt:
                    log(root, f"phase {p.name}: interrupted (Ctrl-C); `ldr run` resumes here")
                    raise
                except (Exception, SystemExit) as e:  # noqa: BLE001  a phase module may sys.exit
                    msg = f"{type(e).__name__}: {str(e)[:300]}"
                    if p.optional:
                        log(root, f"phase {p.name}: optional, failed ({msg}); going on")
                        continue
                    log(root, f"phase {p.name}: FAILED ({msg}); fix the cause and run `ldr run` again to resume")
                    return 1
                log(root, f"phase {p.name}: done in {time.monotonic() - t0:.0f}s")
                if p.name == until:
                    log(root, f"stopped after {until} (--until)")
                    return 0
            if until is None:
                touch(root / ".final_done")
                log(root, f"run done: {root / 'REPORT.md'}")
            return 0
        finally:
            current[0] = "idle"
            switch.close()
            stop_sampler()


def start_sampler(root, phase):
    """telemetry samples every 30 s in a daemon thread while the run lasts (never fails the run) -> stop()"""
    try:
        from . import telemetry
    except Exception:  # noqa: BLE001  telemetry is optional
        return lambda: None
    ev = threading.Event()

    def loop():
        while not ev.is_set():
            with contextlib.suppress(Exception):
                telemetry.sample_once(root, phase())
            ev.wait(30)
    threading.Thread(target=loop, name="telemetry", daemon=True).start()
    return ev.set


def status(root, plist=None):
    """[(phase, done?, optional?, about)] from the files on disk"""
    root = research_dir(root)
    out = []
    for p in phases() if plist is None else plist:
        try:
            ok = bool(p.done(root))
        except Exception:  # noqa: BLE001  a half-written file must not break `ldr status`
            ok = False
        out.append((p.name, ok, p.optional, p.about))
    return out


def status_text(root, tail=8):
    root = research_dir(root)
    rows = status(root)
    todo = [n for n, ok, _, _ in rows if not ok]
    state = ("finished" if (root / ".final_done").exists() else "in progress") + (
        f"; next phase: {todo[0]}" + (f" (not done: {', '.join(todo)})" if len(todo) > 1 else "") if todo else "")
    lines = [f"topic: {root.parent}", f"state: {state}"]
    try:
        from .notes import profile
        p = profile(root)
        if p:
            lines.append(f"profile: {p.get('profile')} rounds {p.get('rounds')} x {p.get('pages')} pages, "
                         f"output language {p.get('out_lang', 'en')}")
    except Exception:  # noqa: BLE001
        pass
    lines.append("")
    lines += [f"  [{'x' if ok else ' '}] {n:<17} {about}{' (optional)' if opt else ''}" for n, ok, opt, about in rows]
    logf = root / ".run.log"
    if logf.exists():
        lines += ["", "last log lines:"] + ["  " + x for x in
                                            logf.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]]
    return "\n".join(lines)
