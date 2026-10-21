"""Shared plumbing for the script-style suites (tests/suite_*.py).

Each suite is a plain script: it runs its checks at module level, prints one PASS/FAIL line per check and ends with
"N passed, M failed" (exit 1 on any failure). tests/test_suites.py runs every suite in its own interpreter, so module
globals a suite monkeypatches (corpus.CACHE, harvest.LANES, outline.llm ...) never leak into another suite.

Run one directly with `python tests/suite_<name>.py [scratch_dir]`. The package is used from the installed copy
(`pip install -e .`); a plain checkout without install also works because src/ is put on the path below.
"""
import atexit
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Hermetic settings, set before any package module loads its config (harvest reads it at import). A developer's own
# ldr.toml / LDR_* / OPENAI_* would otherwise point an unpatched call at a real LLM server or SearXNG, and config.get()
# would ask that server for its model list. 127.0.0.2:9 refuses at once; 127.0.0.1:9 can hang under WSL2 mirrored
# networking (seen live), so it is never used as the "nothing listens here" address.
for _k in [k for k in os.environ if k.startswith(("LDR_", "OPENAI_"))]:
    del os.environ[_k]
_fd, _EMPTY_CFG = tempfile.mkstemp(prefix="ldr-test-", suffix=".toml")
os.close(_fd)
atexit.register(lambda: os.path.exists(_EMPTY_CFG) and os.unlink(_EMPTY_CFG))
os.environ.update({"LDR_CONFIG": _EMPTY_CFG, "LDR_LLM_BASE_URL": "http://127.0.0.2:9/v1", "LDR_LLM_MODEL": "test-model",
                   "LDR_SEARXNG_URL": "http://127.0.0.2:9", "LDR_CACHE_DIR": tempfile.mkdtemp(prefix="ldr-test-cache-"),
                   # embeddings on the CPU model: with a free GPU the larger GPU model scores differently and an
                   # off-topic test cluster crossed the pack similarity floor (results must not depend on the machine)
                   "LDR_EMBED_DEVICE": "cpu", "LDR_EMBED_MODEL_GPU": ""})
atexit.register(shutil.rmtree, os.environ["LDR_CACHE_DIR"], True)

try:
    import local_deep_research
except ModuleNotFoundError:  # a checkout that was never `pip install -e .`-ed: use ../src
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    import local_deep_research

PKG_DIR = pathlib.Path(local_deep_research.__file__).resolve().parent  # the package's source files
SRC_ROOT = PKG_DIR.parent                                               # the dir that holds the package
REPO = pathlib.Path(__file__).resolve().parent.parent                   # the checkout (integrations/, docs/ ...)
_OWN_SCRATCH = False


def scratch(prefix):
    """A fresh scratch dir: argv[1] when given (kept afterwards for inspection), else a new temp dir that finish()
    removes when every check passed (kept on failure so the files behind a FAIL can be read)."""
    global _OWN_SCRATCH
    if len(sys.argv) > 1:
        p = pathlib.Path(sys.argv[1])
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)
        return p
    _OWN_SCRATCH = True
    return pathlib.Path(tempfile.mkdtemp(prefix=prefix))


def child_env(**extra):
    """Environment for a child interpreter: the package importable even when not installed, UTF-8 pipes (Windows
    pipes default to the ANSI code page and the CLIs print Turkish / CJK text), no user config file picked up."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **extra}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC_ROOT), os.environ.get("PYTHONPATH", "")) if p)
    return env


def run_module(module, *args, timeout=120, cwd=None):
    """`python -m local_deep_research.<module> args...` -> (returncode, stdout + stderr).

    The old suites ran scripts with `python -I` so a planted module in the cwd could not be imported. -I would also drop
    PYTHONPATH (and so the package of an uninstalled checkout); running from a temp cwd keeps the same guarantee."""
    r = subprocess.run([sys.executable, "-m", f"local_deep_research.{module}", *map(str, args)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout, env=child_env(),
                       cwd=cwd or tempfile.gettempdir())
    return r.returncode, r.stdout + r.stderr


def can_symlink(where):
    """True when this process may create symlinks (Windows needs developer mode or admin; the escape checks that
    rely on them are skipped there instead of failing)."""
    probe = pathlib.Path(where) / ".symlink-probe"
    try:
        os.symlink(where, probe)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


def finish(results, scratch_dir):
    """print the summary line the harness looks for and exit 1 on any failure"""
    passed, failed = results
    keep = failed or not _OWN_SCRATCH
    print(f"\n{passed} passed, {failed} failed" + (f"  (scratch: {scratch_dir})" if keep else ""))
    if not keep:
        shutil.rmtree(scratch_dir, ignore_errors=True)
    sys.exit(1 if failed else 0)
