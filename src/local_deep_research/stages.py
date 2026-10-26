"""Per-stage models: the collecting phases and the report-writing phases may use different models and servers.

  collect = profile ... numeric   (plan, search, extraction, verification, notes, gap round): many short calls
  report  = outline ... telemetry (outline, analysis, writing, report verification): fewer, long calls

Measured why (2026-10-26, 2x RTX 3090, the same Vienna-1683 question): a fast MoE model (Sage-A1.5, 35B-A3B)
collected more verified evidence 3-5x faster than a dense 27B model, while the dense model wrote the better report
from the same evidence. The hybrid gave the best report in under half the dense model's time.

Settings ([collect] / [report] in ldr.toml, see config.py): every key left empty inherits [llm]. When the two models
do not fit on the GPUs together, start_cmd / stop_cmd let ldr swap them: entering a stage runs the other stage's
stop_cmd, then this stage's start_cmd, then waits until the server lists a model (stage_ready_timeout). Commands are
split like a shell line but run without a shell (use a script for pipes or &&; on Windows the line goes to
CreateProcess as written). The switch happens only when a phase
of the stage actually runs, so resuming a run whose collecting is done never starts the collecting model.
"""
import dataclasses
import shlex
import subprocess
import sys
import time

from . import config

REPORT_FROM = "outline"  # the first report phase; every phase before it collects


def stage_of(phase, order):
    """"collect" or "report" for a phase name, given the run's phase names in order"""
    order = list(order)
    if REPORT_FROM in order and phase in order and order.index(phase) >= order.index(REPORT_FROM):
        return "report"
    return "collect"


def configured(s):
    """True when any stage setting differs from plain [llm] (otherwise the switcher does nothing)"""
    return any(config.stage_overrides(s, st) or any(config.stage_cmds(s, st)) for st in config.STAGES)


def run_cmd(cmd, what, log):
    if not cmd:
        return
    log(f"stage: {what}: {cmd}")
    # Windows: CreateProcess parses the command line itself (shlex in non-POSIX mode would keep the quotes)
    args = cmd if sys.platform == "win32" else shlex.split(cmd)
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"{what} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")


def wait_ready(s, timeout, log, poll=10):
    """the stage server's model id once it answers /models (the configured one must be listed); raises on timeout"""
    t0 = time.monotonic()
    while True:
        ids = config.list_models(s.llm_base_url, s.llm_api_key)
        if ids and (not s.llm_model or s.llm_model in ids):
            return s.llm_model or ids[0]
        if time.monotonic() - t0 > timeout:
            want = f"model {s.llm_model!r}" if s.llm_model else "any model"
            raise RuntimeError(f"{s.llm_base_url} did not list {want} within {timeout}s (listed: {ids[:5]})")
        time.sleep(max(0.5, min(poll, timeout - (time.monotonic() - t0))))


class Switcher:
    """installs each stage's settings as the process-wide config when a phase of that stage starts"""

    def __init__(self, log):
        self.base = config.get()
        self.written = config.written_model()  # get() replaced an empty llm_model with the [llm] server's first
        self.log = log
        self.active = None
        self.on = configured(self.base)

    def enter(self, stage):
        if not self.on or stage == self.active:
            return
        s = config.stage_settings(self.base, stage)
        s.llm_model = config.stage_settings(dataclasses.replace(self.base, llm_model=self.written), stage).llm_model
        start, _ = config.stage_cmds(self.base, stage)
        if self.active:
            run_cmd(config.stage_cmds(self.base, self.active)[1], f"{self.active} stop_cmd", self.log)
        run_cmd(start, f"{stage} start_cmd", self.log)
        if start or not s.llm_model:  # a just-started server (or an unnamed model) is waited for / resolved
            s.llm_model = wait_ready(s, int(self.base.stage_ready_timeout or 1800), self.log)
        config.install(s)
        reset_limits()
        self.active = stage
        self.log(f"stage {stage}: model {s.llm_model!r} at {s.llm_base_url} (concurrency {s.llm_concurrency}, "
                 f"context {s.llm_context})")

    def close(self):
        """back to the base settings (the server of the last stage keeps running)"""
        if self.on:
            config.install(self.base)
            reset_limits()


def reset_limits():
    """the llm module's request semaphore is sized from llm_concurrency on first use: rebuild it for the new stage"""
    from . import llm
    with llm._sem_lock:
        llm._sem = None
