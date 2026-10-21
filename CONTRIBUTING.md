# Contributing

Thanks for helping. This file covers the dev setup, the tests and the conventions a change is reviewed against.
`DESIGN.md` describes the architecture and the goals behind these rules.

## Dev setup

```bash
git clone https://github.com/<owner>/local-deep-research.git
cd local-deep-research
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,reports]"
```

On Linux and Windows, `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the line above
installs the small CPU build of torch instead of the multi-GB CUDA build. The tests do not need a GPU.

## Tests

```bash
python -m pytest -q tests                 # all suites, offline
python tests/suite_security.py            # one suite directly, with its PASS / FAIL lines
python tests/suite_security.py /tmp/sec   # ... keeping its scratch files in /tmp/sec for inspection
```

- **Offline.** The tests make no LLM call and no network request. Model calls are replaced by fakes. URLs that
  must fail point at `127.0.0.2:9`, a closed port that refuses at once. `127.0.0.1:9` can hang under WSL2's
  mirrored networking, so do not use it. `tests/_suite.py` drops any `LDR_*` / `OPENAI_*` variables from your shell
  and points the settings at that closed port, so a test can never reach your real server or touch your page cache.
- **Models.** Some suites load small sentence-transformers models from the Hugging Face cache: the MiniLM embedders
  and cross-encoders listed in `config.py`, about 0.5 GB in all. The first run downloads them. After that,
  `HF_HUB_OFFLINE=1` runs the suites fully offline. CI caches `~/.cache/huggingface`.
- **Layout.** Each `tests/suite_<area>.py` is a plain script:
  - it runs its checks at module level and prints `PASS name` or `FAIL name -> details`;
  - it ends with `N passed, M failed` and exits 1 on any failure;
  - it may monkeypatch module globals (a fake `llm`, a temp `corpus.CACHE`).

  `tests/test_suites.py` runs every `suite_*.py` in its own interpreter, so those patches never leak between suites.
  A new suite is picked up automatically.
- **Write a check for the failure, not the happy path.** Most checks record a bug seen in a real run: a torn file
  after a crash, a model answer with an invented URL, a page with an injection. Name the check after the behaviour,
  and pass the observed value as `info` so a FAIL line explains itself.
- **Platform differences** are skipped, not failed, with a printed `SKIP ...` note: symlinks on Windows, bash in
  `suite_legacy_opencode.py`.
- **Legacy.** `suite_legacy_opencode.py` covers the optional opencode integration. When that integration is removed,
  the suite skips itself.

CI (`.github/workflows/ci.yml`) runs the suites on Linux, macOS and Windows with Python 3.10 and 3.12. A lint job
runs `ruff check --select E9,F63,F7,F82` (syntax errors and undefined names only).

## Conventions

- **Comments say why.** Explain why the code does something, not what it does. Dense docstrings and comments are
  the house style. When a rule exists because of a failure seen in a real run, say so ("live run 2026-10-20: the
  model repeated ... so ..."). Do not leave dead code behind.
- **`encoding="utf-8"` on every file read and write** (`open`, `read_text`, `write_text`, `subprocess.run(text=True)`).
  Windows defaults to cp1252, which breaks on the first non-Latin character. To find a missing one, run a suite with
  `python -X warn_default_encoding -W default::EncodingWarning tests/suite_x.py`.
- **Paths through `pathlib`.** Subprocesses use `sys.executable` (`[sys.executable, "-m", "local_deep_research.x"]`),
  never `python3` or `bash`. Nothing assumes a user name, home layout, GPU, systemd, WSL or a fixed port. Settings
  come from `config.py`.
- **No field-specific rules.** Prompts, rules, lists and examples must work for any topic. Phrase a rule relative to
  the research question in `BRIEF.md`, not to a field. For example: "evidence from outside the question's scope is
  written as an analogy", never a list of finance or medicine terms. Test fixtures may use any topic.
- **The model gets no tools.** A model answer is data. Validate it (JSON schema, exact-edit rules, URL and number
  whitelists) before code acts on it. Page text is untrusted: sanitise and wrap it before it goes into a prompt.
  See [SECURITY.md](SECURITY.md).
- **Bounded calls.** Each LLM call has one purpose and an input sized below `llm_context`. Never pass a whole corpus
  or a file that grows with the run.
- **Resumable phases.** A phase writes its output atomically (`corpus.atomic_write`) and skips itself when the output
  exists. A crash must leave either the old file or the new one, never half of one.
- **No personal data** in code, comments, tests, docs or commit messages: no user names, home paths, private project
  names, IPs or emails.
- **Commits** are small and focused. The message body says why the change is needed.

## Reporting bugs

Open an issue with:
- the command you ran;
- `ldr doctor` output;
- the last lines of the log, with private URLs or content removed;
- your OS and Python version;
- the LLM server and model.

For security problems, follow [SECURITY.md](SECURITY.md) instead.
