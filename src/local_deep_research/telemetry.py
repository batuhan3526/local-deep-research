"""Run telemetry for a deep-research run: tokens, server and GPU performance, energy, per-phase and per-call use.

  python -m local_deep_research.telemetry sample <topic-dir> [--interval 30] [--opencode-db PATH]
        sampler loop for a run driven from outside (`ldr run` samples in-process, see pipeline.start_sampler): every
        interval one row in research/telemetry/samples.csv: server counters (vLLM-style Prometheus /metrics: prompt /
        generated tokens, requests, KV cache, prefix-cache hits, latency sums), every NVIDIA GPU (utilization, power,
        memory, temperature, SM clock), host load / memory, and the phase. Stops (after a last report) once
        research/.final_done exists.
  python -m local_deep_research.telemetry report <topic-dir> [--opencode-db PATH]
        research/TELEMETRY.md + research/telemetry/summary.json: totals, rates, energy (power integrated over the
        samples), tokens / minutes / energy per phase, the model calls per kind from dataset/llm/*.jsonl (works with
        any OpenAI-compatible server: it reads the "usage" each answer reports), the opencode token use per agent
        when an opencode database is given (optional agent mode), and the corpus sizes the run produced.

Everything outside the run's files is optional and degrades to empty columns: a server without /metrics, a machine
without nvidia-smi, an OS without /proc or os.getloadavg (Windows). The metrics URL is the setting metrics_url, else
<llm_base_url without /v1>/metrics. Server counters are cumulative since the server started: the report uses deltas
between samples (a server restart = a counter reset, handled), so tokens spent before the sampler started are not
in the per-phase table; the report says so. Read-only towards everything but research/telemetry/ and TELEMETRY.md.
"""
import argparse
import collections
import contextlib
import csv
import datetime
import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import time

OPENCODE_DB = None  # optional agent mode: path of opencode.db (--opencode-db / setting opencode_db); None = none
VLLM_KEYS = {  # column -> metric name (summed over label sets)
    "prompt_tok": "vllm:prompt_tokens_total", "gen_tok": "vllm:generation_tokens_total",
    "req_ok": "vllm:request_success_total", "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting", "kv_usage": "vllm:kv_cache_usage_perc",
    "pc_queries": "vllm:prefix_cache_queries_total", "pc_hits": "vllm:prefix_cache_hits_total",
    "preempt": "vllm:num_preemptions_total", "e2e_sum": "vllm:e2e_request_latency_seconds_sum",
    "e2e_n": "vllm:e2e_request_latency_seconds_count", "ttft_sum": "vllm:time_to_first_token_seconds_sum",
    "ttft_n": "vllm:time_to_first_token_seconds_count"}
COUNTERS = ("prompt_tok", "gen_tok", "req_ok", "pc_queries", "pc_hits", "preempt", "e2e_sum", "e2e_n", "ttft_sum",
            "ttft_n")
GPU_FIELDS = ("util", "power_w", "limit_w", "mem_mib", "temp_c", "sm_mhz")
_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eEnaNinf]+)$")


def _settings():
    from . import config
    return config.load()  # file + env only: telemetry never asks the server for its model list


def metrics_url():
    s = _settings()
    return getattr(s, "metrics_url", "") or re.sub(r"/v1/?$", "", s.llm_base_url.rstrip("/")) + "/metrics"


def opencode_db():
    """the opencode database to read agent sessions from, or None (the core pipeline has no agents)"""
    p = OPENCODE_DB or getattr(_settings(), "opencode_db", "")
    return pathlib.Path(p).expanduser() if p and pathlib.Path(p).expanduser().exists() else None


def vllm_metrics():
    try:
        import httpx
        from . import corpus
        txt = httpx.get(metrics_url(), timeout=10, trust_env=False, headers=corpus.auth()).text
    except Exception:  # noqa: BLE001  no /metrics: empty columns
        return {}
    acc = collections.defaultdict(float)
    for line in txt.splitlines():
        m = _LINE.match(line.strip())
        if m:
            acc[m.group(1)] += float(m.group(3))
    return {k: acc.get(v, "") for k, v in VLLM_KEYS.items()}


def gpus():
    if not shutil.which("nvidia-smi"):
        return []
    q = "index,utilization.gpu,power.draw,power.limit,memory.used,temperature.gpu,clocks.sm"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=15).stdout
    except Exception:  # noqa: BLE001
        return []
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def phase(root, current=""):
    """the pipeline phase (+ 'r<round>:<stage>' while harvesting) and active opencode agent types if a database is
    configured (sessions of this topic updated in the last 90 s)"""
    bits = []
    if current in ("harvest", "gap"):
        try:
            st = json.loads((root / "harvest/state.json").read_text(encoding="utf-8"))
            bits.append(f"harvest r{st.get('round')}:{st.get('stage')}" if current == "harvest" else "gap-round")
        except (OSError, ValueError):
            bits.append(current)
    elif current and current not in ("idle", "start"):
        bits.append(current)
    db = opencode_db()
    if db:
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            cut = int((time.time() - 90) * 1000)
            rows = c.execute("select title, parent_id from session where directory = ? and time_updated > ?",
                             (str(root.parent), cut)).fetchall()
            c.close()
            kinds = collections.Counter(agent_kind(t, p) for t, p in rows)
            if kinds:
                bits.append("agents " + " ".join(f"{k}x{n}" for k, n in sorted(kinds.items())))
        except sqlite3.Error:
            pass
    return "; ".join(bits) or "idle"


def agent_kind(title, parent):
    m = re.search(r"@(\w+) subagent", title or "")
    return m.group(1) if m else ("orchestrator" if not parent else "subagent")


def host():
    """(load average 1 min, used RAM GiB); '' where the OS does not tell (Windows has no getloadavg / /proc)"""
    load = ram = ""
    with contextlib.suppress(Exception):
        load = round(os.getloadavg()[0], 2)
    with contextlib.suppress(Exception):
        with open("/proc/meminfo", encoding="utf-8") as f:
            mem = {ln.split(":")[0]: int(ln.split()[1]) for ln in f if ln.split(":")[0] in ("MemTotal", "MemAvailable")}
        ram = round((mem["MemTotal"] - mem["MemAvailable"]) / 2 ** 20, 2)
    return load, ram


def sample_once(root, current=""):
    """append one row to research/telemetry/samples.csv (the header is fixed by the first sample)"""
    out = root / "telemetry"; out.mkdir(exist_ok=True)
    f = out / "samples.csv"
    g = gpus()
    if not f.exists():
        head = ["ts", "phase", *VLLM_KEYS, *[f"gpu{gi[0]}_{k}" for gi in g for k in GPU_FIELDS], "load1", "ram_gib"]
        with open(f, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(head)
    with open(f, newline="", encoding="utf-8") as fh:
        head = next(csv.reader(fh), [])
    load, ram = host()
    row = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "phase": phase(root, current),
           **vllm_metrics(), "load1": load, "ram_gib": ram}
    for gi in g:
        row.update({f"gpu{gi[0]}_{k}": x for k, x in zip(GPU_FIELDS, gi[1:])})
    with open(f, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=head, extrasaction="ignore").writerow(row)


def cmd_sample(a):
    root = a.root
    last_report = 0.0
    while True:
        sample_once(root, "external")
        done = (root / ".final_done").exists()
        if done or time.time() - last_report > 1800:  # a fresh TELEMETRY.md every 30 min, and a last one at the end
            try:
                report(root, quiet=True)
            except Exception as e:  # noqa: BLE001
                print(f"report failed: {e}", file=sys.stderr)
            last_report = time.time()
        if done:
            print("run finished: last report written, sampler stops")
            return
        time.sleep(a.interval)


def phase_key(ph):
    """'harvest r2:extract; agents researcherx7 skepticx1' -> 'harvest r2 extract + researcher, skeptic'; a pipeline
    phase ('notes', 'write') stays as it is: what shares the server at the same time is attributed together
    (agent counts dropped)."""
    parts = [x.strip() for x in (ph or "idle").split(";") if x.strip() not in ("", "idle")]
    h = next((re.sub(r"^harvest r(\d+):(\w+)", r"harvest r\1 \2", x) for x in parts if x.startswith(("harvest", "gap"))), "")
    ag = next((x for x in parts if x.startswith("agents ")), "")
    other = next((x for x in parts if x != ag and not x.startswith(("harvest", "gap"))), "")
    kinds = ", ".join(sorted({re.sub(r"x\d+$", "", k) for k in ag[7:].split()})) if ag else ""
    return " + ".join(x for x in (h or other, kinds) if x) or "idle"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def opencode_tokens(root):
    """{agent kind: {sessions, messages, input, output, reasoning, cache_read, cache_write}} for this topic."""
    out = collections.defaultdict(lambda: collections.Counter())
    db = opencode_db()
    if db is None:
        return {}
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    sess = {sid: agent_kind(t, p) for sid, t, p in
            c.execute("select id, title, parent_id from session where directory = ?", (str(root.parent),))}
    for sid, data in c.execute("select session_id, data from message"):
        if sid not in sess:
            continue
        try:
            d = json.loads(data)
        except ValueError:
            continue
        tk = d.get("tokens") if isinstance(d, dict) else None
        if not isinstance(tk, dict):
            continue
        k = sess[sid]
        o = out[k]
        o["messages"] += 1
        for name in ("input", "output", "reasoning"):
            o[name] += int(tk.get(name) or 0)
        cache = tk.get("cache") or {}
        o["cache_read"] += int(cache.get("read") or 0); o["cache_write"] += int(cache.get("write") or 0)
    for k in set(sess.values()):
        out[k]["sessions"] = sum(1 for v in sess.values() if v == k)
    c.close()
    return {k: dict(v) for k, v in out.items()}


def corpus_sizes(root):
    s = {}
    try:
        with open(root / "harvest/sources.csv", newline="", encoding="utf-8") as f:
            s["pages_with_text"] = sum(1 for _ in csv.DictReader(f))
    except OSError:
        pass
    try:
        with open(root / "claims/extracted.jsonl", "rb") as f:
            s["claims_extracted"] = sum(1 for _ in f)
    except OSError:
        pass
    try:
        csv.field_size_limit(1 << 24)
        with open(root / "claims/claims.csv", newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        s["clusters"] = len(rows)
        s.update({f"clusters_{k}": sum(r.get("confidence") == k for r in rows) for k in ("Yüksek", "Orta", "Düşük")})
    except OSError:
        pass
    s["rounds"] = len(list((root / "harvest").glob("round-*.json")))
    rep = root / "REPORT.md"
    if rep.exists():
        s["report_words"] = len(rep.read_text(encoding="utf-8", errors="replace").split())
    return s


def llm_calls(root):
    """{kind: {calls, prompt_tokens, completion_tokens, latency_s, cut}} from dataset/llm/<kind>.jsonl (llmlog): the
    token use of every model call of the run, from the usage each answer reports (any OpenAI-compatible server)"""
    out = {}
    for f in sorted((root / "dataset/llm").glob("*.jsonl")) if (root / "dataset/llm").is_dir() else []:
        c = collections.Counter()
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                u = r.get("usage") or {}
                c["calls"] += 1
                c["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                c["completion_tokens"] += int(u.get("completion_tokens") or 0)
                c["latency_s"] += float(r.get("latency_s") or 0)
                c["cut"] += r.get("finish_reason") == "length"
        out[f.stem] = {k: round(v, 1) if k == "latency_s" else v for k, v in c.items()}
    return out


def report(root, quiet=False):
    f = root / "telemetry/samples.csv"
    rows = []
    if f.exists():
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    per = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    gpu_cols = sorted({c.rsplit("_", 1)[0] for c in (rows[0] if rows else {}) if c.startswith("gpu") and c.endswith("_util")})
    util_sum, n_util, peak_run = collections.Counter(), 0, 0.0
    for prev, cur in zip(rows, rows[1:]):
        t0 = datetime.datetime.fromisoformat(prev["ts"]); t1 = datetime.datetime.fromisoformat(cur["ts"])
        dt = (t1 - t0).total_seconds()
        if dt <= 0 or dt > 600:  # sampler gap (crash / reboot): not attributed
            continue
        ph = phase_key(cur["phase"])
        d = {}
        for k in COUNTERS:
            a_, b_ = fnum(prev.get(k)), fnum(cur.get(k))
            d[k] = (b_ - a_ if b_ >= a_ else b_) if a_ is not None and b_ is not None else 0.0  # reset -> since 0
        watts = sum(fnum(cur.get(f"{g}_power_w")) or 0 for g in gpu_cols)
        for target in (per[ph], tot):
            target["seconds"] += dt
            target["kwh"] += watts * dt / 3.6e6
            for k in COUNTERS:
                target[k] += d[k]
        for g in gpu_cols:
            util_sum[g] += fnum(cur.get(f"{g}_util")) or 0
        n_util += 1
        peak_run = max(peak_run, fnum(cur.get("running")) or 0)
    last = rows[-1] if rows else {}
    life = {k: fnum(last.get(k)) for k in ("prompt_tok", "gen_tok", "req_ok")}
    first = rows[0] if rows else {}
    before = {k: fnum(first.get(k)) for k in ("prompt_tok", "gen_tok", "req_ok")}  # spent before the sampler started
    oc = opencode_tokens(root)
    calls = llm_calls(root)
    cs = corpus_sizes(root)
    rate = lambda c: (c["gen_tok"] / c["seconds"]) if c.get("seconds") else 0.0  # noqa: E731
    summ = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "samples": len(rows), "from": rows[0]["ts"] if rows else None, "to": last.get("ts"),
            "totals": {k: round(v, 3) for k, v in tot.items()},
            "vllm_lifetime_counters_at_last_sample": life,
            "gpu_util_avg": {g: round(util_sum[g] / n_util, 1) for g in gpu_cols} if n_util else {},
            "peak_concurrent_requests": peak_run, "per_phase": {k: {x: round(y, 3) for x, y in v.items()} for k, v in per.items()},
            "llm_calls_per_kind": calls, "opencode_tokens_per_agent": oc, "corpus": cs}
    out = root / "telemetry"; out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    h = tot["seconds"] / 3600
    md = ["# Run telemetry", "", f"Samples {len(rows)} ({summ['from']} → {summ['to']}). Server counters (/metrics) are cumulative since "
          "the server started; the tables use deltas between samples, so tokens spent before the sampler started are "
          "only in the lifetime line.", "",
          "## Totals (sampled period)", "", "| metric | value |", "|---|---|",
          f"| wall time | {h:.2f} h |", f"| prompt tokens (input) | {tot['prompt_tok']:,.0f} |",
          f"| generated tokens (output) | {tot['gen_tok']:,.0f} |", f"| requests finished | {tot['req_ok']:,.0f} |",
          f"| generation throughput (avg) | {rate(tot):.1f} tok/s |",
          f"| prompt throughput (avg) | {(tot['prompt_tok'] / tot['seconds'] if tot['seconds'] else 0):.1f} tok/s |",
          f"| prefix-cache hit rate | {(100 * tot['pc_hits'] / tot['pc_queries'] if tot['pc_queries'] else 0):.1f}% |",
          f"| mean request latency | {(tot['e2e_sum'] / tot['e2e_n'] if tot['e2e_n'] else 0):.1f} s |",
          f"| mean time to first token | {(tot['ttft_sum'] / tot['ttft_n'] if tot['ttft_n'] else 0):.2f} s |",
          f"| peak concurrent requests | {peak_run:.0f} |", f"| preemptions | {tot['preempt']:.0f} |",
          f"| GPU energy | {tot['kwh']:.2f} kWh |",
          *[f"| {g} utilization (avg) | {v}% |" for g, v in summ["gpu_util_avg"].items()],
          f"| server lifetime counters (since its start) | prompt {life['prompt_tok'] or 0:,.0f} / generated "
          f"{life['gen_tok'] or 0:,.0f} / requests {life['req_ok'] or 0:,.0f} |",
          f"| before the sampler started (server counters at the first sample, since its start; not split by "
          f"phase) | prompt {before['prompt_tok'] or 0:,.0f} / generated {before['gen_tok'] or 0:,.0f} / requests "
          f"{before['req_ok'] or 0:,.0f} |", "",
          "## Per phase", "", "| phase | minutes | prompt tokens | generated tokens | gen tok/s | requests | kWh |",
          "|---|---|---|---|---|---|---|"]
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]["seconds"]):
        md.append(f"| {k} | {v['seconds'] / 60:.1f} | {v['prompt_tok']:,.0f} | {v['gen_tok']:,.0f} | {rate(v):.1f} | "
                  f"{v['req_ok']:,.0f} | {v['kwh']:.2f} |")
    md += ["", "## Model calls per kind (dataset/llm)", "", "| kind | calls | prompt tokens | completion tokens | "
           "mean latency | cut at max_tokens |", "|---|---|---|---|---|---|"]
    for k, v in sorted(calls.items(), key=lambda kv: -kv[1].get("prompt_tokens", 0)):
        md.append(f"| {k} | {v.get('calls', 0):,} | {v.get('prompt_tokens', 0):,} | {v.get('completion_tokens', 0):,} | "
                  f"{v.get('latency_s', 0) / max(1, v.get('calls', 0)):.1f} s | {v.get('cut', 0)} |")
    if oc:
        md += ["", "## Agent token use (opencode)", "", "| agent | sessions | messages | input | output | reasoning | "
               "cache read |", "|---|---|---|---|---|---|---|"]
        for k, v in sorted(oc.items(), key=lambda kv: -(kv[1].get("input", 0) + kv[1].get("output", 0))):
            md.append(f"| {k} | {v.get('sessions', 0)} | {v.get('messages', 0)} | {v.get('input', 0):,} | "
                      f"{v.get('output', 0):,} | {v.get('reasoning', 0):,} | {v.get('cache_read', 0):,} |")
    md += ["", "## Corpus produced", "", "| item | count |", "|---|---|", *[f"| {k} | {v:,} |" for k, v in cs.items()], ""]
    (root / "TELEMETRY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if not quiet:
        print(f"-> {root / 'TELEMETRY.md'}, {out / 'summary.json'} ({len(rows)} samples)")


def main(argv=None):
    global OPENCODE_DB
    ap = argparse.ArgumentParser(description="run telemetry: sampler loop / TELEMETRY.md report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "report"):
        p = sub.add_parser(name); p.add_argument("root", type=pathlib.Path)
        p.add_argument("--opencode-db", type=pathlib.Path, help="optional agent mode: opencode.db to read sessions from")
        if name == "sample":
            p.add_argument("--interval", type=int, default=30)
    a = ap.parse_args(argv)
    OPENCODE_DB = a.opencode_db or OPENCODE_DB
    a.root = a.root.resolve()
    if a.root.name != "research" and (a.root / "research").is_dir():
        a.root = a.root / "research"
    cmd_sample(a) if a.cmd == "sample" else report(a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
