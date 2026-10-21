"""Offline checks for config.py: defaults, TOML sections, env overrides, bool / int coercion, the misplaced-key
rescue, and that config.example.toml documents every setting with its default (a new setting must be added there).
usage: python tests/suite_config.py"""
import dataclasses
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from local_deep_research import config  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


tmp = pathlib.Path(tempfile.mkdtemp(prefix="cfg-test-"))
ex = HERE.parent / "config.example.toml"
data = tomllib.loads(ex.read_text(encoding="utf-8"))
keys = set()
for k, v in data.items():
    keys |= {f"{k}_{kk}" for kk in v} if isinstance(v, dict) else {k}
fields = [f.name for f in dataclasses.fields(config.Settings)]
check("config.example.toml documents every setting", not [f for f in fields if f not in keys],
      [f for f in fields if f not in keys])
check("config.example.toml loads to exactly the defaults", config.load(ex, env={}) == config.Settings())
f = tmp / "c.toml"
f.write_text('[llm]\nbase_url = "http://h:1/v1"\nno_think = true\ncontext = 65536\nsearxng_url = "http://s:2"\n',
             encoding="utf-8")
s = config.load(f, env={})
check("sections flatten, bool / int kept, a top-level key misplaced inside [llm] is rescued",
      s.llm_base_url == "http://h:1/v1" and s.llm_no_think is True and s.llm_context == 65536
      and s.searxng_url == "http://s:2", s)
s = config.load(f, env={"LDR_LLM_NO_THINK": "0", "LDR_LLM_CONTEXT": "8192", "OPENAI_API_KEY": "k"})
check("env wins over the file; bool text '0' = False; OPENAI_API_KEY as fallback key",
      s.llm_no_think is False and s.llm_context == 8192 and s.llm_api_key == "k", s)
s = config.load(f, env={"OPENAI_BASE_URL": "http://o/v1", "LDR_LLM_BASE_URL": "http://l/v1"})
check("LDR_LLM_BASE_URL beats OPENAI_BASE_URL", s.llm_base_url == "http://l/v1", s.llm_base_url)
s = config.Settings(searxng_url="http://x:8888/", llm_base_url="http://y/v1/")
check("derived URLs: /search appended once, chat URL", s.search_url == "http://x:8888/search"
      and config.Settings(searxng_url="http://x/search").search_url == "http://x/search"
      and s.chat_url == "http://y/v1/chat/completions", (s.search_url, s.chat_url))
check("list_models: unreachable server -> [] (closed port refuses at once)",
      config.list_models("http://127.0.0.2:9/v1", timeout=3) == [])

print(f"\n{RESULTS[0]} passed, {RESULTS[1]} failed")
sys.exit(1 if RESULTS[1] else 0)
