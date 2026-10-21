# Security

## Threat model

local-deep-research reads large amounts of text written by strangers and passes it to an LLM. It runs on your
machine, usually next to other local services: the LLM server, SearXNG, perhaps databases or admin panels on the LAN.
The design assumes the following.

**Untrusted:**
- every fetched web page, PDF, search result and API response;
- every answer of the LLM, because a page in its prompt may have steered it;
- local files listed in `BRIEF.md`, which are read as data, never as instructions.

**Trusted:**
- your configuration (`ldr.toml`, `LDR_*` variables);
- the LLM server and SearXNG you point it at;
- the Python environment.

**What an attacker controls:** the content of pages the harvest fetches. This includes text aimed at the model
("ignore previous instructions"), hidden HTML, crafted PDFs, redirects, very large or compressed responses, and
DNS answers for their own domains.

**What they should not be able to do:**
- make `ldr` request internal addresses (SSRF);
- run commands or read or write files outside the topic folder;
- exhaust memory with a crafted document;
- get unsupported URLs or numbers into the report, or raise a claim's confidence with injected text.

## Defences

- **No tools for the model.** In the core pipeline the model can only return text or JSON. It cannot run commands,
  read files or fetch URLs. Code decides every next step.
- **Validated output.**
  - JSON answers are schema-checked, and string sizes and counts are capped.
  - Report edits must quote existing text exactly and uniquely.
  - A URL or number the model introduces must already be in the evidence, otherwise the edit is rejected or the URL
    is removed.
- **Prompt hygiene.**
  - Page text is cleaned of chat-template tokens (`<|im_start|>`, `[INST]`, `<think>` ...), role prefixes and
    invisible or bidi characters.
  - Hidden HTML (`display:none`, `aria-hidden`, comments, same-colour text) is removed.
  - The text is wrapped in random per-request markers and sent as data, separate from the system instructions.
  - Pages that look like prompt injection are flagged. They cannot corroborate a claim or raise its confidence, and
    their follow-up questions are dropped.
- **SSRF guards.**
  - Only public addresses are fetched. Loopback, private, link-local, CGNAT, metadata, IPv4-mapped IPv6 and odd
    numeric forms of an IP are refused.
  - These checks also run after each redirect and on every resolved address (DNS rebinding). The connection is
    pinned to the checked IP while the `Host` header and TLS SNI keep the name.
  - Only `http`/`https` URLs are fetched, and redirects, response size (after decompression) and fetch time are
    capped.
- **Parsers.** PDFs are parsed in a separate process with memory and CPU limits and a timeout. Where the OS offers no
  such limits, the timeout and size caps still apply.
- **Files.** Paths built from model or page text must resolve inside the topic folder (`corpus.under`). Symlinks that
  lead outside it are refused. Outputs are written atomically.
- **SQL.** All SQL uses parameters. A test fails on f-string or concatenated SQL.

The offline test suite (`tests/suite_security.py` and others) exercises these defences, including private literals,
DNS rebinding, redirects to `file://` and to internal ports, injection strings, malformed JSON and symlink escapes.

## Your part

- Keep SearXNG and the LLM server bound to `127.0.0.1` (the provided `docker/searxng` setup does this), or protect
  them with authentication. The SearXNG setup has its rate limiter off and must not be exposed.
- Treat reports as research material. They quote web content, and a confident tone is not proof.
- Do not list secrets, credentials or private keys in `BRIEF.md`'s local files. Their text goes into prompts and
  outputs.
- Downloaded datasets (`dataset/`) contain raw web text. Open them as data, not with tools that execute content.

## Out of scope

- The security of your LLM server, SearXNG, or the websites and APIs that are queried.
- Model jailbreaks that only change wording within the evidence. They are a quality problem; please open a regular
  issue.
- Optional integrations under `integrations/`, such as agent harnesses that give a model tools. They follow their
  own permission model; please report problems in them all the same.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability. Use GitHub's private vulnerability reporting instead
("Security" tab of the repository, then "Report a vulnerability").

Include:
- the affected version or commit;
- a minimal reproduction, for example a page or PDF that triggers the problem (served locally);
- the impact you observed.

We aim to acknowledge reports within a week and to fix confirmed issues in the next release. Reporters are credited
unless they prefer not to be.
