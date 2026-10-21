"""Settings: which LLM server, model, search backend and folders a run uses. Nothing here is tied to one machine.

Resolution order (later wins):
  1. defaults below
  2. a TOML file: $LDR_CONFIG, else ./ldr.toml, else the user config dir (Linux ~/.config/local-deep-research/
     config.toml, macOS ~/Library/Application Support/local-deep-research/config.toml, Windows %APPDATA%\\
     local-deep-research\\config.toml)
  3. environment variables: LDR_<KEY> (e.g. LDR_LLM_BASE_URL); OPENAI_BASE_URL / OPENAI_API_KEY are honoured as
     fallbacks so any OpenAI-compatible server works (vLLM, llama.cpp server, Ollama, LM Studio, TGI, SGLang ...)
An empty llm_model means: the first model the server lists at /v1/models.
"""
import dataclasses
import json
import os
import pathlib
import sys
import urllib.request

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

APP = "local-deep-research"


def user_dir(kind):
    """per-user config / data / cache dir for this OS (no platformdirs dependency)"""
    home = pathlib.Path.home()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA" if kind == "cache" else "APPDATA") or str(home / "AppData" / "Roaming")
        return pathlib.Path(base) / APP / ("cache" if kind == "cache" else "")
    if sys.platform == "darwin":
        sub = {"config": "Library/Application Support", "data": "Library/Application Support", "cache": "Library/Caches"}
        return home / sub[kind] / APP
    env = {"config": "XDG_CONFIG_HOME", "data": "XDG_DATA_HOME", "cache": "XDG_CACHE_HOME"}[kind]
    dflt = {"config": ".config", "data": ".local/share", "cache": ".cache"}[kind]
    return pathlib.Path(os.environ.get(env) or home / dflt) / APP


@dataclasses.dataclass
class Settings:
    # LLM: any OpenAI-compatible chat completions server
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = ""                  # "" = first id from /v1/models
    llm_api_key: str = ""                # sent as "Authorization: Bearer"; empty = no header
    llm_context: int = 32768             # the server's max context (tokens); prompts are sized below it
    llm_timeout: int = 900               # seconds per call (long answers on a slow server need minutes)
    llm_retries: int = 3
    llm_no_think: bool = False           # send chat_template_kwargs.enable_thinking=false (Qwen3-style models
                                         # otherwise think at length on every call; harmless elsewhere)
    llm_json_mode: str = "json_object"   # json_object | none (servers that reject response_format json_object)
    llm_concurrency: int = 16            # parallel requests the server handles well: ~1-4 llama.cpp / Ollama,
                                         # 32-64 vLLM / SGLang on a big GPU
    # optional second (smaller / faster) model for bulk claim extraction and judging; empty = the main one
    extract_base_url: str = ""
    extract_model: str = ""
    extract_api_key: str = ""
    # search: a SearXNG instance with the JSON format enabled (see docker/searxng); scholarly APIs need no key
    searxng_url: str = "http://localhost:8888"
    # share of each round's selected pages reserved for scholarly sources (journals, DOI / arXiv, scholarly
    # encyclopedias, university repositories) when enough relevant ones are found; 0 = no reservation
    academic_share: float = 0.4
    # local embedding / reranking models (sentence-transformers, downloaded on first use)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model_gpu: str = "BAAI/bge-m3"  # used only when a CUDA GPU has room; "" = never
    multilingual_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_device: str = "auto"           # auto | cpu | cuda | cuda:N | mps (auto: a GPU only when it has room)
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    multilingual_cross_encoder: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # telemetry / optional integrations
    metrics_url: str = ""                # Prometheus /metrics of the LLM server; "" = derived from llm_base_url
    opencode_db: str = ""                # only for the optional opencode agent mode (integrations/opencode)
    # folders
    cache_dir: str = ""                  # "" = user cache dir (page cache shared by all research)
    research_root: str = ""              # "" = ./research-projects (where `ldr new` creates topics)

    @property
    def chat_url(self):
        return self.llm_base_url.rstrip("/") + "/chat/completions"

    @property
    def extract_url(self):
        return (self.extract_base_url or self.llm_base_url).rstrip("/") + "/chat/completions"

    @property
    def search_url(self):
        u = self.searxng_url.rstrip("/")
        return u if u.endswith("/search") else u + "/search"

    def cache_path(self):
        return pathlib.Path(self.cache_dir).expanduser() if self.cache_dir else user_dir("cache")

    def projects_path(self):
        return pathlib.Path(self.research_root).expanduser() if self.research_root else pathlib.Path.cwd() / "research-projects"


def config_file():
    for p in (os.environ.get("LDR_CONFIG"), "ldr.toml", user_dir("config") / "config.toml"):
        if p and pathlib.Path(p).expanduser().is_file():
            return pathlib.Path(p).expanduser()
    return None


def _coerce(field, value):
    t = field.type if isinstance(field.type, type) else {"int": int, "str": str, "bool": bool,
                                                         "float": float}.get(str(field.type), str)
    if t is bool:  # TOML gives a bool; env vars give text ("1", "true", "yes", "on")
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    return int(value) if t is int else float(value) if t is float else str(value)


def load(path=None, env=None):
    """-> Settings from defaults, the TOML file and the environment"""
    env = os.environ if env is None else env
    s = Settings()
    fields = {f.name: f for f in dataclasses.fields(Settings)}
    f = pathlib.Path(path) if path else config_file()
    if f:
        data = tomllib.loads(f.read_text(encoding="utf-8"))
        flat = {}
        for k, v in data.items():  # [llm] base_url = ... and llm_base_url = ... both work
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if f"{k}_{kk}" not in fields and kk in fields:
                        # TOML puts every key after "[llm]" into that table: a top-level setting written below a
                        # section header is the most common config mistake, so accept it and say how to fix it
                        print(f"config: {kk!r} is inside [{k}] in {f}; it is used as the top-level setting "
                              f"(move it above the first [section] header)", file=sys.stderr)
                        flat[kk] = vv
                    else:
                        flat[f"{k}_{kk}"] = vv
            else:
                flat[k] = v
        for k, v in flat.items():
            if k in fields:
                setattr(s, k, _coerce(fields[k], v))
            else:
                print(f"config: unknown key {k!r} in {f} (ignored)", file=sys.stderr)
    if not env.get("LDR_LLM_BASE_URL") and env.get("OPENAI_BASE_URL"):
        s.llm_base_url = env["OPENAI_BASE_URL"]
    if not env.get("LDR_LLM_API_KEY") and env.get("OPENAI_API_KEY"):
        s.llm_api_key = env["OPENAI_API_KEY"]
    for name, fl in fields.items():
        v = env.get("LDR_" + name.upper())
        if v not in (None, ""):
            setattr(s, name, _coerce(fl, v))
    return s


def list_models(base_url, api_key="", timeout=10):
    """ids the server offers at /v1/models ([] when unreachable)"""
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - user-configured local endpoint
            return [m.get("id") for m in json.loads(r.read().decode()).get("data", []) if m.get("id")]
    except Exception:  # noqa: BLE001
        return []


_SETTINGS = None


def get():
    """the process-wide settings (loaded once; the model id resolved from the server when left empty)"""
    global _SETTINGS
    if _SETTINGS is None:
        s = load()
        if not s.llm_model:
            ids = list_models(s.llm_base_url, s.llm_api_key)
            s.llm_model = ids[0] if ids else "default"
        if (s.extract_base_url or s.extract_api_key) and not s.extract_model:
            ids = list_models(s.extract_base_url or s.llm_base_url, s.extract_api_key or s.llm_api_key)
            s.extract_model = ids[0] if ids else s.llm_model
        _SETTINGS = s
    return _SETTINGS


def set_for_tests(s):
    """install explicit settings (tests, embedding callers); None resets to lazy loading"""
    global _SETTINGS
    _SETTINGS = s
