"""Offline checks for per-stage models and API compatibility: [collect] / [report] settings, the phase -> stage map,
the stage switch inside pipeline.run (start / stop commands, the stage's model and concurrency seen by its phases, a
resumed run that never starts the collecting model), wire() for APIs that want max_completion_tokens or no
temperature, and the resend of throttled (429) answers. Two tiny OpenAI-style stub servers stand in for the models.
usage: python tests/suite_stages.py [scratch_dir]"""
import http.server
import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _suite import finish, scratch  # noqa: E402

from local_deep_research import config, corpus, pipeline, stages  # noqa: E402

RESULTS = [0, 0]
TMP = scratch("stages-test-")


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


def stub(model, throttle=0):
    """an OpenAI-style server listing `model`; the first `throttle` chat calls get 429 + Retry-After: 1"""
    seen = {"bodies": [], "left": throttle}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, headers=()):
            data = json.dumps(obj).encode()
            self.send_response(code)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"data": [{"id": model}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["bodies"].append(body)
            if seen["left"] > 0:
                seen["left"] -= 1
                return self._send(429, {"error": "slow down"}, [("Retry-After", "1")])
            self._send(200, {"choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]})

    for host in ("127.0.0.2", "127.0.0.1"):  # 127.0.0.1 can stall under WSL2 mirrored networking; macOS lacks .2
        try:
            srv = http.server.ThreadingHTTPServer((host, 0), H)
            break
        except OSError:
            continue
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://{host}:{srv.server_address[1]}/v1", seen


# ---------- settings ----------
f = TMP / "c.toml"
f.write_text('[llm]\nbase_url = "http://a:1/v1"\nmodel = "dense"\napi_key = "k"\ncontext = 65536\n'
             '[collect]\nmodel = "fast"\nconcurrency = 64\nno_think = true\n'
             '[report]\nbase_url = "http://b:2/v1"\ncontext = 32768\n', encoding="utf-8")
s = config.load(f, env={})
c, r = config.stage_settings(s, "collect"), config.stage_settings(s, "report")
check("[collect] overrides typed like [llm] (int, bool) and inherits the rest",
      c.llm_model == "fast" and c.llm_concurrency == 64 and c.llm_no_think is True and c.llm_context == 65536
      and c.llm_base_url == "http://a:1/v1" and c.llm_api_key == "k", c)
check("[report] on another server: no model / key taken from [llm] (resolved from that server)",
      r.llm_base_url == "http://b:2/v1" and r.llm_model == "" and r.llm_api_key == "" and r.llm_context == 32768, r)
check("plain settings: no stage switching", not stages.configured(config.Settings()) and stages.configured(s))
order = [p.name for p in pipeline.phases()]
check("phase -> stage: everything before outline collects, outline and later report",
      [stages.stage_of(n, order) for n in ("plan", "harvest", "gap", "numeric", "outline", "write", "verify")]
      == ["collect"] * 4 + ["report"] * 3)

# ---------- wire / throttling ----------
config.set_for_tests(config.Settings(llm_token_param="max_completion_tokens", llm_temperature=False))
b = corpus.wire({"model": "m", "max_tokens": 9, "temperature": 0.2, "messages": []})
check("wire: max_completion_tokens and no temperature for APIs that refuse them",
      b == {"model": "m", "max_completion_tokens": 9, "messages": []}, b)
config.set_for_tests(config.Settings())
b = corpus.wire({"model": "m", "max_tokens": 9, "temperature": 0.2})
check("wire: the default body is sent unchanged", b == {"model": "m", "max_tokens": 9, "temperature": 0.2}, b)


class R:
    def __init__(self, ra):
        self.headers = {"retry-after": ra} if ra is not None else {}


check("throttle_wait: Retry-After seconds, else 5 / 10 / 20 s backoff, capped",
      corpus.throttle_wait(R("3"), 0) == 3 and corpus.throttle_wait(R(None), 0) == 5
      and corpus.throttle_wait(R(None), 2) == 20 and corpus.throttle_wait(R("9999"), 0) == 120)
url, seen = stub("m", throttle=1)
config.set_for_tests(config.Settings(llm_base_url=url, llm_model="m", llm_token_param="max_completion_tokens"))
rr = corpus.send(url + "/chat/completions", {"model": "m", "max_tokens": 5, "messages": []}, {}, 30)
check("send: a 429 is resent after Retry-After and the resent body is wired",
      rr.status_code == 200 and len(seen["bodies"]) == 2 and seen["bodies"][-1].get("max_completion_tokens") == 5
      and "max_tokens" not in seen["bodies"][-1], seen["bodies"])

# ---------- the switch inside pipeline.run ----------
url_a, _ = stub("fast-moe")
url_b, _ = stub("dense-writer")
cmdlog = TMP / "cmds.txt"
py = sys.executable.replace("\\", "/")


def cmd(word):
    return f'"{py}" -c "open(r\'{cmdlog}\', \'a\').write(\'{word}\\n\')"'


def topic(name):
    root = TMP / name
    root.mkdir()
    (root / "BRIEF.md").write_text("# BRIEF: test\n", encoding="utf-8")
    return root


seen_by = {}


def mk(name):
    def run(ctx):
        st = config.get()
        seen_by[name] = (st.llm_model, ctx.conc, st.llm_base_url)
        (ctx.root / f".{name}").touch()
    return pipeline.Phase(name, run, lambda root, n=name: (root / f".{n}").exists())


plist = [mk(n) for n in ("profile", "plan", "harvest", "numeric", "outline", "write", "verify")]
base = config.Settings(llm_base_url=url_b, llm_model="dense-writer", llm_concurrency=16,
                       collect_base_url=url_a, collect_concurrency="64", collect_start_cmd=cmd("start-collect"),
                       collect_stop_cmd=cmd("stop-collect"), report_start_cmd=cmd("start-report"),
                       stage_ready_timeout=30)
config.set_for_tests(base)
root = topic("hybrid")
code = pipeline.run(root, plist=plist, sample=False)
cmds = cmdlog.read_text(encoding="utf-8").split() if cmdlog.exists() else []
check("hybrid run: exit 0, commands in order start-collect, stop-collect, start-report",
      code == 0 and cmds == ["start-collect", "stop-collect", "start-report"], (code, cmds))
check("collect phases used the collect server's model (resolved from its /models) with concurrency 64",
      all(seen_by[n][:2] == ("fast-moe", 64) and seen_by[n][2] == url_a for n in ("plan", "harvest", "numeric")),
      seen_by)
check("report phases used [llm]'s model and concurrency",
      all(seen_by[n][:2] == ("dense-writer", 16) and seen_by[n][2] == url_b for n in ("outline", "write", "verify")),
      seen_by)
check("after the run the base settings are back", config.get() is base)
log = (root / ".run.log").read_text(encoding="utf-8")
check("the run log names each stage's model", "stage collect: model 'fast-moe'" in log
      and "stage report: model 'dense-writer'" in log, log[-600:])

cmdlog.unlink()
seen_by.clear()
root2 = topic("resumed")
for n in ("profile", "plan", "harvest", "numeric"):
    (root2 / f".{n}").touch()
code = pipeline.run(root2, plist=plist, sample=False)
cmds = cmdlog.read_text(encoding="utf-8").split() if cmdlog.exists() else []
check("resume after collecting: only the report server is started", code == 0 and cmds == ["start-report"]
      and set(seen_by) == {"outline", "write", "verify"}, (code, cmds, seen_by))

cmdlog.unlink()
config.set_for_tests(config.Settings(llm_base_url="http://127.0.0.2:9/v1", llm_model="x",
                                     report_start_cmd=cmd("start-report"), stage_ready_timeout=1))
root3 = topic("down")
for n in ("profile", "plan", "harvest", "numeric"):
    (root3 / f".{n}").touch()
code = pipeline.run(root3, plist=plist, sample=False)
log = (root3 / ".run.log").read_text(encoding="utf-8")
check("a stage server that never comes up fails the run with a clear log line (resumable)",
      code == 1 and "stage switch" in log and "did not list" in log, log[-400:])

config.set_for_tests(None)
finish(RESULTS, TMP)
