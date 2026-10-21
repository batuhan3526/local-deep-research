"""pytest entry for the script-style suites: every tests/suite_*.py (present and future) runs in its own interpreter
and must exit 0.

Why subprocesses: the suites monkeypatch module globals (corpus.CACHE, corpus.llm_judge, harvest.LANES, outline.llm
...) at module level. Importing them into one pytest process would let one suite's patches leak into the next, and a
suite that calls sys.exit() would end the session. One interpreter per suite keeps them as independent as when they
are run by hand (`python tests/suite_<name>.py`).
"""
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SUITES = sorted(HERE.glob("suite_*.py"))
TIMEOUT = int(os.environ.get("LDR_SUITE_TIMEOUT", "1800"))  # s per suite; the slowest takes ~1-2 min on a laptop CPU


@pytest.mark.parametrize("suite", SUITES, ids=[p.stem for p in SUITES])
def test_suite(suite, tmp_path):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # the suites print Turkish / CJK; Windows pipes default to cp1252
    r = subprocess.run([sys.executable, str(suite), str(tmp_path / "scratch")], cwd=tmp_path, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
    out = (r.stdout + r.stderr).splitlines()
    fails = [line for line in out if line.startswith("FAIL ")]
    tail = "\n".join(out[-60:])
    assert r.returncode == 0, (f"{suite.name} exited {r.returncode}\n--- FAIL lines ---\n" + "\n".join(fails[:40])
                               + f"\n--- last {min(60, len(out))} lines ---\n{tail}")


def test_suites_found():
    """a renamed or moved tests dir must not turn into a silent 'nothing ran'"""
    assert len(SUITES) >= 7, SUITES
