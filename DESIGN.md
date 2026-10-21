# local-deep-research — design

A local deep-research tool. The user brings their own OpenAI-compatible LLM server and a SearXNG instance.
The tool harvests web and scholarly evidence, verifies every claim against its source text and writes a long,
cited report (Markdown, plus PDF / HTML / XLSX / PPTX when available).

## Goals

1. **Any topic.** The user can research anything: medicine, the history of wars, art, software, finance ...
   - No rule, prompt, list or example may assume a field.
   - Rules are phrased relative to the research question in `BRIEF.md`. Example: "evidence from outside the
     question's scope (another field, population, period or setting) is written as an analogy". The model judges
     scope in each call.
   - There is no per-topic configuration and no "domain" setting.
2. **Any machine.**
   - Runs on Linux, macOS and Windows with Python 3.10+.
   - Nothing assumes a user name, home layout, GPU, systemd, bash, WSL or a fixed port.
   - Endpoints, model, API key, SearXNG URL and folders come from `config.py`: a TOML file (`ldr.toml` or the user
     config dir) and `LDR_*` env vars. `OPENAI_BASE_URL` / `OPENAI_API_KEY` are fallbacks.
3. **Any user level.**
   - One command: `ldr run "question"`.
   - Sensible defaults.
   - Clear errors when the LLM server or SearXNG is unreachable (with the URL it tried and how to fix it).
   - Resumable after a crash or Ctrl-C.
4. **Deterministic orchestration by code.**
   - A small local model (~27B, 32-64k context) cannot drive a multi-step agent run reliably. Live runs showed that
     agents which read several big files overflowed, compacted, lost the task and wrote nothing.
   - So every phase is code. Each code phase makes **bounded, single-purpose LLM calls**: one input set, one output
     file, no tools.
   - opencode (or any agent harness) is an OPTIONAL integration, never required.
5. **Safety stays.**
   - Web text is untrusted data.
   - The model never gets tools in the core pipeline.
   - Its answers are validated (JSON schema / exact-edit rules). URLs and numbers it adds must exist in the evidence.
   - SSRF guards, the injection screen and capped parser subprocesses stay.

## Pipeline (`ldr run`, all resumable from files under `<topic>/research/`)

| phase | code | output |
|---|---|---|
| 0 brief | cli (`ldr new` writes BRIEF.md from the question; the user may edit it) | BRIEF.md |
| 1 plan + harvest | harvest.py plan / run (rounds, languages, dynamic stop) | harvest/, claims/, corpus |
| 2 reader notes | notes.py: one call per sub-question over its first pack file (bounded) | notes/s<n>.md |
| 3 gap round | harvest.py gap (uses the notes' gaps) | more claims |
| 4 analysis | corpus contradictions, numeric_findings, outline.py, sections.py analyze | analysis/ |
| 5 synthesis + writing | synthesis.py: STYLE.md + ANALYSIS.md (from the Özet heads), sections.py write, per-part counter-arguments, executive summary, contradictions section; assemble_report | report/, REPORT.md |
| 6 verification | verify_fix.py, check_report.py | VERIFY-*.md, CHECK.md |
| 7 outputs | appendix, md2pdf (HTML fallback), export_extras, dataset, telemetry | REPORT.pdf/html, APPENDIX, dataset.zip |

Optional local project context:
- BRIEF.md may list local files or folders the research should take into account.
- notes.py reads them (bounded) into notes/local-project.md, and local_facts.py quotes their numbers verbatim.
- With no local files, these steps are skipped.

## Package layout

```
src/local_deep_research/   the package (relative imports only: `from . import corpus`)
  config.py                settings (done)
  cli.py, pipeline.py      `ldr` entry point and the phase runner          (agent B)
  llm.py                   one OpenAI-compatible client: chat_text / chat_json, retries, logging   (agent B)
  notes.py, synthesis.py   code versions of the former agent phases       (agent B)
  ... existing modules
tests/                     offline tests (no LLM, no network), runnable with `pytest`   (agent D)
integrations/opencode/     optional agent mode                            (agent B)
integrations/systemd/      optional Linux service templates               (agent B)
docker/searxng/            compose file + settings with the JSON format on (agent D)
```

## Conventions

- Match the existing style:
  - dense docstrings that say WHY;
  - comments that record the live failure behind a rule;
  - no dead code.
- Every file read or write passes `encoding="utf-8"`. Windows defaults to cp1252.
- Paths go through `pathlib`. Subprocesses use `sys.executable`, never `bash` or `python3`.
- No personal data: user names, home paths, private project names, IPs or emails. A comment may say
  "live run 2026-10-20", never a private project name.
- Commits are small and focused, each with a body that says why.
