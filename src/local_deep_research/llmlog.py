"""Raw LLM call log for a research run (v3.1): every request the harvest / corpus code sends to the model and the
model's raw answer, one JSON line per call, in research/dataset/llm/<kind>.jsonl.

Row: {"ts", "kind", "model", "messages" (the exact chat sent), "params" (temperature, max_tokens, response_format),
"response" (raw text the model returned), "finish_reason", "usage" {prompt_tokens, completion_tokens}, "latency_s",
"meta" (url / chunk / sub-question ...)}. "messages" + {"role": "assistant", "content": response} is a ready
chat-format fine-tuning example; "kind" says which task (extract, screen, judge, review, plan, gap, lang, summary,
classify). Never raises: logging must not break a run. Untrusted web text inside the prompts is kept as sent.
"""
import datetime
import json
import os
import pathlib
import re
import threading

_lock = threading.Lock()
_dir = None


def set_root(root):
    """Log into <research root>/dataset/llm/ from now on (None disables)."""
    global _dir
    _dir = (pathlib.Path(root) / "dataset/llm") if root else None


def record(kind, body, resp, latency, meta=None):
    """body: the request JSON sent to /v1/chat/completions; resp: the parsed response JSON (or None)."""
    if _dir is None:
        return
    try:
        ch = ((resp or {}).get("choices") or [{}])[0]
        row = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "kind": kind, "model": body.get("model"),
               "messages": body.get("messages"),
               "params": {k: body.get(k) for k in ("temperature", "max_tokens", "response_format") if k in body},
               "response": (ch.get("message") or {}).get("content"), "finish_reason": ch.get("finish_reason"),
               "usage": {k: (resp or {}).get("usage", {}).get(k) for k in ("prompt_tokens", "completion_tokens")},
               "latency_s": round(float(latency), 3), "meta": meta or {}}
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _lock:
            _dir.mkdir(parents=True, exist_ok=True)
            # file name from a fixed word list, but kept to [\w.-] so no caller can make it invalid on Windows;
            # errors="replace": a lone surrogate in page text must not lose the row; newline: LF on every OS
            name = re.sub(r"[^\w.-]", "_", str(kind))[:60] or "other"
            with open(_dir / f"{name}.jsonl", "a", encoding="utf-8", errors="replace", newline="\n") as f:
                f.write(line)
    except Exception:  # noqa: BLE001  logging never breaks the run
        pass


if os.environ.get("RESEARCH_LLM_LOG_ROOT"):
    set_root(os.environ["RESEARCH_LLM_LOG_ROOT"])
