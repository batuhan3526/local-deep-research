# SearXNG for local-deep-research

[SearXNG](https://docs.searxng.org/) is the metasearch engine `ldr` uses for web results. Scholarly sources
(arXiv, Semantic Scholar, Crossref and others) are queried directly and need no setup.

## Start

```bash
cd docker/searxng
# 1. put a random string into settings.yml -> server.secret_key, for example:
python -c "import secrets; print(secrets.token_hex(32))"
# 2. start it
docker compose up -d
# 3. check: this must print JSON, not an HTML page or "403 Forbidden"
curl "http://localhost:8888/search?q=test&format=json"
```

`ldr` uses `http://localhost:8888` by default (`searxng_url` in `ldr.toml`, or `LDR_SEARXNG_URL`).
`ldr doctor` checks that the instance answers and that the JSON format is on.

## What this setup changes from SearXNG's defaults

| setting | value | why |
|---|---|---|
| `search.formats` | `[html, json]` | `ldr` reads JSON; without it every query fails with 403 |
| `server.limiter` | `false` | one research run sends hundreds of queries from one client; the limiter would block it |
| port | `127.0.0.1:8888` | with the limiter off the instance must stay private to this machine |
| `server.secret_key` | placeholder | replace it; SearXNG will not start with its stock key |

## Already running SearXNG elsewhere?

Point `searxng_url` at it and make sure its `settings.yml` lists `json` under `search.formats`. If it is shared
with other people, keep its limiter on and expect some queries to be throttled during large runs.

## Podman / no Docker

`podman compose up -d` works with the same file. Without containers, follow SearXNG's
[installation guide](https://docs.searxng.org/admin/installation.html) and apply the same three settings.
