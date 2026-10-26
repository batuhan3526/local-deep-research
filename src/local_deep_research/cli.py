"""`ldr`: the command line of local-deep-research.

  ldr doctor                      check the LLM server, SearXNG, the embedding model and the folders; print fixes
  ldr new "question" [--dir D] [--name SLUG] [--no-draft]
                                  create <projects>/<slug>/research/BRIEF.md (edit it before `ldr run` if you like)
  ldr run <topic | "question"> [--profile normal|derin|maksimum|dinamik] [--langs de,zh|all|none] [--out-lang en]
          [--max-rounds N] [--sources N] [--until PHASE]
                                  run (or resume) the whole pipeline; a question creates the topic first
  ldr status <topic>              which phases are done, the next one, the last log lines
Settings (server URL, model, key, SearXNG URL, folders) come from ldr.toml / the user config file / LDR_* env vars
(config.py); --config FILE picks a file. Errors name the URL that was tried and the setting that changes it.
Imports stay light here (`ldr --help` must not load numpy / torch): the pipeline modules load when a command runs.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import unicodedata

from . import __version__, pipeline

PROFILES = ("normal", "derin", "maksimum", "dinamik", "fast", "deep", "max", "dynamic")
PHASES = [p.name for p in pipeline.phases()]
FIX_LLM = ("start your OpenAI-compatible server (vLLM, llama.cpp, Ollama, LM Studio ...), or point ldr at it: "
           "llm_base_url = \"http://HOST:PORT/v1\" in ldr.toml, or the env var LDR_LLM_BASE_URL (OPENAI_BASE_URL works "
           "too)")
FIX_KEY = "set the server's API key: llm_api_key in ldr.toml or LDR_LLM_API_KEY (OPENAI_API_KEY works too)"
FIX_SEARX = ("start SearXNG (a ready setup is in the repository's docker/searxng folder) or point ldr at yours: "
             "searxng_url in ldr.toml or LDR_SEARXNG_URL")
FIX_JSON = "enable the JSON output format in SearXNG's settings.yml: search: formats: [html, json] (see docker/searxng)"
BRIEF = """# BRIEF: {title}

## Question
{question}

## Scope
{scope}

## Local files
<!-- optional: files or folders the research should take into account, one per line as `path` (text files only;
     nothing outside the listed paths is read) -->

## Output language
<!-- optional: a language code such as en, tr, de (`ldr run --out-lang` overrides it) -->
"""
SCOPE_HINT = ("<!-- optional: what is in and out of scope (period, population, setting, kinds of sources to prefer "
              "or avoid) and what the result is for -->")
SCOPE_PROMPT = """Research question: {q}

Draft the scope section of a research brief for this question as 4-8 short Markdown bullets: the sub-areas the
research should cover, what is out of scope, the time period that matters, and the kinds of sources that fit.
Output only the bullets, in the language of the question."""


class UserError(Exception):
    """a problem the user can fix; printed without a traceback"""


def settings():
    from . import config
    return config.load()  # file + env only: no request to the server just to read settings


# ---------- doctor ----------
def _get(url, key="", timeout=5, params=None):
    import httpx
    return httpx.get(url, params=params, timeout=timeout, trust_env=False,
                     headers={"Authorization": f"Bearer {key}"} if key else {})


def check_llm(s, test_call=True, timeout=5):
    """[(name, ok, message, fix)] for the chat server: reachable, the model served, one short answer"""
    url = s.llm_base_url.rstrip("/") + "/models"
    try:
        r = _get(url, s.llm_api_key, timeout)
    except Exception as e:  # noqa: BLE001  any transport error means the same to the user
        return [("LLM server", False, f"cannot reach {url} ({type(e).__name__}: {str(e)[:120]})", FIX_LLM)]
    if r.status_code in (401, 403):
        return [("LLM server", False, f"{url} answered HTTP {r.status_code} (API key missing or wrong)", FIX_KEY)]
    try:
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
    except (ValueError, AttributeError):
        ids = []
    if r.status_code != 200 or not ids:
        return [("LLM server", False, f"{url} answered HTTP {r.status_code} without a model list",
                 "check that llm_base_url ends with the API root (usually /v1)")]
    out = [("LLM server", True, f"{s.llm_base_url} serves {', '.join(ids[:5])}", "")]
    model = s.llm_model or ids[0]
    if s.llm_model and s.llm_model not in ids:
        return out + [("model", False, f"model {s.llm_model!r} is not served (available: {', '.join(ids[:8])})",
                       "set llm_model in ldr.toml / LDR_LLM_MODEL to one of them, or leave it empty for the first")]
    if not test_call:
        return out
    import httpx
    from . import corpus
    try:
        r = httpx.post(s.chat_url, timeout=120, trust_env=False,
                       headers={"Authorization": f"Bearer {s.llm_api_key}"} if s.llm_api_key else {},
                       json=corpus.wire({"model": model, "max_tokens": 16,
                                         "messages": [{"role": "user", "content": "Say OK."}]}, s))
        r.raise_for_status()
        r.json()["choices"][0]
    except Exception as e:  # noqa: BLE001
        return out + [("test call", False, f"{s.chat_url} with model {model!r}: {type(e).__name__}: {str(e)[:160]}",
                       "check the server log; the model must accept /chat/completions requests")]
    return out + [("test call", True, f"model {model!r} answered a short request", "")]


def check_models(s, test_call=True, need=("collect", "report")):
    """check_llm for every server the run uses: [llm] alone, or each stage's ([collect] / [report]) when stage
    settings are set. A stage with a start_cmd is not checked: ldr starts its server when the stage begins."""
    from . import config, stages
    if not stages.configured(s):
        return check_llm(s, test_call)
    out, seen = [], set()
    for st in need:
        ss = config.stage_settings(s, st)
        if config.stage_cmds(s, st)[0]:
            out.append((f"{st} server", True, f"{ss.llm_base_url}: started by {st}_start_cmd when the stage begins",
                        ""))
            continue
        key = (ss.llm_base_url.rstrip("/"), ss.llm_model, ss.llm_api_key)
        if key in seen:
            continue
        seen.add(key)
        out += [(f"{st}: {n}", ok, m, fix) for n, ok, m, fix in check_llm(ss, test_call)]
    return out


def check_searx(s, timeout=10):
    url = s.search_url
    try:
        r = _get(url, timeout=timeout, params={"q": "test", "format": "json"})
    except Exception as e:  # noqa: BLE001
        return [("SearXNG", False, f"cannot reach {url} ({type(e).__name__}: {str(e)[:120]})", FIX_SEARX)]
    if r.status_code == 403:
        return [("SearXNG", False, f"{url} refused format=json (HTTP 403)", FIX_JSON)]
    try:
        n = len(r.json().get("results", []))
    except (ValueError, AttributeError):
        return [("SearXNG", False, f"{url} answered HTTP {r.status_code} without JSON", FIX_JSON)]
    return [("SearXNG", True, f"{url} answered with {n} results", "")]


def check_embed(s):
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(s.embed_model, device="cpu").encode(["test"])
    except Exception as e:  # noqa: BLE001
        return [("embedding model", False, f"{s.embed_model}: {type(e).__name__}: {str(e)[:160]}",
                 "pip install sentence-transformers; the model downloads from Hugging Face on first use (network "
                 "needed once), or set embed_model to a model in your local HF cache")]
    return [("embedding model", True, f"{s.embed_model} loads on CPU", "")]


def check_dirs(s):
    out = []
    for name, d in (("cache dir", s.cache_path()), ("projects dir", s.projects_path())):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".ldr-write-test"
            probe.write_text("ok", encoding="utf-8"); probe.unlink()
            out.append((name, True, str(d), ""))
        except OSError as e:
            key = "cache_dir" if name == "cache dir" else "research_root"
            out.append((name, False, f"{d} is not writable ({e})", f"set {key} in ldr.toml / LDR_{key.upper()}"))
    return out


def report(checks):
    for name, ok, msg, fix in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {msg}")
        if not ok and fix:
            print(f"       fix: {fix}")
    return all(c[1] for c in checks)


def cmd_doctor(a):
    from . import config
    s = settings()
    f = config.config_file()
    print(f"config: {f or 'none found (defaults + environment)'}")
    checks = check_models(s) + check_searx(s)
    if not a.skip_embed:
        checks += check_embed(s)
    return 0 if report(checks + check_dirs(s)) else 1


# ---------- new ----------
def slugify(text, cap=60):
    s = unicodedata.normalize("NFKD", text.replace("ı", "i")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    if len(s) > cap:
        s = s[:cap].rsplit("-", 1)[0]
    return s or "topic"


def draft_scope(s, question):
    """one bounded call to draft the scope bullets, only when the server answers its model list within 3 s"""
    if not check_llm(s, test_call=False, timeout=3)[-1][1]:
        return None
    from . import llm
    try:
        out = llm.chat_text(SCOPE_PROMPT.format(q=question), 600, "brief-scope", tries=1)
    except llm.LLMError:
        return None
    bullets = [ln.rstrip() for ln in out.splitlines() if re.match(r"^\s*[-*]\s+\S", ln)]
    return "\n".join(bullets[:10]) or None


def create_topic(question, base=None, name=None, draft=True):
    """-> topic folder with research/BRIEF.md. The same question again reuses its folder (so `ldr run "question"`
    resumes); another question under an existing name is refused."""
    question = " ".join(str(question).split())
    if len(question) < 8:
        raise UserError("the question is too short; write it as a full sentence")
    s = settings()
    topic = pathlib.Path(base).expanduser() if base else s.projects_path()
    topic = (topic / (name and slugify(name) or slugify(question))).resolve()
    brief = topic / "research/BRIEF.md"
    if brief.exists():
        if f"\n{question}\n" in brief.read_text(encoding="utf-8", errors="replace"):
            return topic
        raise UserError(f"{topic} already holds another topic; pick a name with --name")
    scope = draft_scope(s, question) if draft else None
    brief.parent.mkdir(parents=True, exist_ok=True)
    brief.write_text(BRIEF.format(title=question[:120], question=question, scope=(
        "<!-- drafted by the model from the question; edit freely -->\n" + scope) if scope else SCOPE_HINT),
        encoding="utf-8")
    return topic


def cmd_new(a):
    topic = create_topic(a.question, a.dir, a.name, not a.no_draft)
    print(f"topic: {topic}\nbrief: {topic / 'research/BRIEF.md'} (edit it if you like)\nnext:  ldr run {topic}")
    return 0


# ---------- run / status ----------
def find_topic(target):
    """an existing topic / research folder, or a topic name under the projects dir; None otherwise"""
    for p in (pathlib.Path(target).expanduser(), settings().projects_path() / str(target)):
        if p.is_dir() and ((p / "research/BRIEF.md").exists() or (p / "BRIEF.md").exists()):
            return p.resolve()
    return None


def brief_lang(root):
    """the language code under "## Output language" in BRIEF.md, or None"""
    t = (root / "BRIEF.md").read_text(encoding="utf-8", errors="replace") if (root / "BRIEF.md").exists() else ""
    m = re.search(r"(?mi)^#{1,4}\s*(output language|çıktı dili)\s*$\n((?:(?!#).*\n?)*)", t)
    body = re.sub(r"(?s)<!--.*?-->", "", m.group(2)) if m else ""
    code = re.search(r"\b([a-z]{2})\b", body)
    return code.group(1) if code else None


def profile_args(root, a):
    """harvest profile flags for this run: given flags + the existing profile's choices (so `--out-lang tr` on a
    resumed run does not reset its profile or languages); [] keeps harvest/profile.json as it is"""
    f = root / "harvest/profile.json"
    try:
        old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except ValueError:
        old = {}
    given = any(x is not None for x in (a.profile, a.langs, a.out_lang, a.max_rounds, a.sources))
    if f.exists() and not given:
        return []
    out = []
    prof = a.profile or old.get("profile")
    if prof:
        out += ["--profile", prof]
    langs = a.langs or (",".join(old["langs"]) or "none" if isinstance(old.get("langs"), list) else None)
    if langs:
        out += ["--langs", langs]
    ol = a.out_lang or old.get("out_lang") or brief_lang(root)
    if ol:
        out += ["--out-lang", ol]
    if a.max_rounds:
        out += ["--rounds", str(a.max_rounds)]
    if a.sources:
        out += ["--sources", str(a.sources)]
    if old.get("early_stop") is False:
        out.append("--no-early-stop")
    return out


def preflight(root):
    """the servers this run still needs: the LLM always, SearXNG only while the harvest is not done"""
    from . import stages
    s = settings()
    rows = pipeline.status(root)
    order = [n for n, _, _, _ in rows]
    need = tuple(sorted({stages.stage_of(n, order) for n, ok, _, _ in rows if not ok})) or ("report",)
    checks = check_models(s, test_call=False, need=need)
    if not (root / "harvest/.done").exists() or not (root / "harvest/.gap_done").exists():
        checks += check_searx(s)
    bad = [c for c in checks if not c[1]]
    if bad:
        report(bad)
        raise UserError("fix the problems above, then run the same command again (finished phases are kept)")


def cmd_run(a):
    topic = find_topic(a.target)
    if topic is None:
        if re.search(r"[/\\]", a.target) and len(a.target.split()) < 3:
            raise UserError(f"no topic folder {a.target} (a folder needs research/BRIEF.md; or pass a question)")
        topic = create_topic(a.target, a.dir, a.name, not a.no_draft)
        print(f"topic: {topic}")
    root = pipeline.research_dir(topic)
    if not a.no_preflight:
        preflight(root)
    return pipeline.run(root, profile_args(root, a), until=a.until)


def cmd_status(a):
    topic = find_topic(a.topic)
    if topic is None:
        raise UserError(f"no topic folder {a.topic} (needs research/BRIEF.md)")
    print(pipeline.status_text(topic))
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="ldr", description="local deep research: harvest, verify and cite evidence "
                                 "with your own OpenAI-compatible LLM server and SearXNG")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--config", type=pathlib.Path, help="settings file (default: ./ldr.toml, then the user config)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{doctor,new,run,status}")
    p = sub.add_parser("doctor", help="check servers, models and folders; print how to fix each problem")
    p.add_argument("--skip-embed", action="store_true", help="do not load the embedding model")
    p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("new", help="create a topic folder with a BRIEF.md for a question")
    p.add_argument("question")
    p.add_argument("--dir", type=pathlib.Path, help="parent folder (default: research_root / ./research-projects)")
    p.add_argument("--name", help="folder name (default: from the question)")
    p.add_argument("--no-draft", action="store_true", help="no model call to draft the scope")
    p.set_defaults(func=cmd_new)
    p = sub.add_parser("run", help="run or resume the research on a topic folder or a new question")
    p.add_argument("target", help="topic folder, topic name, or a question in quotes")
    p.add_argument("--profile", choices=PROFILES, help="size: normal (~400 sources), derin (~1500-2000), maksimum "
                   "(~5000+), dinamik (until the information dries up); English aliases fast / deep / max / dynamic")
    p.add_argument("--langs", help="extra search languages: de,zh,ja ... | all | none (default: harvest's list)")
    p.add_argument("--out-lang", help="report language code (default: BRIEF.md's, else en)")
    p.add_argument("--max-rounds", type=int, help="harvest rounds (a ceiling for dinamik)")
    p.add_argument("--sources", type=int, help="target number of sources (scales rounds / pages / sub-questions)")
    p.add_argument("--until", choices=PHASES, metavar="PHASE", help="stop after this phase: " + ", ".join(PHASES))
    p.add_argument("--dir", type=pathlib.Path, help="parent folder for a new topic")
    p.add_argument("--name", help="folder name for a new topic")
    p.add_argument("--no-draft", action="store_true", help="no model call to draft a new topic's scope")
    p.add_argument("--no-preflight", action="store_true", help="skip the server reachability checks")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("status", help="phases done / next / last log lines of a topic")
    p.add_argument("topic")
    p.set_defaults(func=cmd_status)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.config:
        os.environ["LDR_CONFIG"] = str(a.config.expanduser().resolve())
    try:
        return a.func(a)
    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; the same command resumes from the last finished phase", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
