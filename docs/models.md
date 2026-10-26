# Models: one model, two models (hybrid), or a hosted API

`ldr` talks to any server that speaks the OpenAI chat-completions protocol. A run has two stages, and each may use
its own model:

| stage | phases | calls |
|---|---|---|
| **collect** | profile, plan, harvest (search, claim extraction, corroboration), contradictions, notes, gap round, numeric | thousands of short calls; throughput matters |
| **report** | outline, analyze, style, write, parts, verify, check, appendix, pdf | tens of long calls; writing quality matters |

Set `[collect]` and/or `[report]` in `ldr.toml`. Every key left empty inherits `[llm]`. Without them, `[llm]` does
everything, as before.

## Measured: Sage-A1.5 vs a dense 27B model, and the hybrid

Hardware: 2x RTX 3090 (24 GB, power limit 55% = 192.5 W each), vLLM 0.28, WSL2. Models:

- **Sage-A1.5** (InternScience; Qwen3.5-MoE, 35B total / 3B active parameters, hybrid linear + full attention),
  4-bit W4A16 AutoRound checkpoint `stuchapin/Sage-A1.5-W4A16-AutoRound` (~18.4 GiB of weights);
- **Qwen3.8-27B** dense, 4-bit W4A16.

### Server throughput (512-token answers, total tok/s)

| parallel requests | Sage TP=2 | Sage DP=2 | Sage TP=2 + MTP 2 | Qwen3.8-27B TP=2 |
|---|---|---|---|---|
| 1 | **75** | 62 | 44 | 36 |
| 16 | **916** | 844 | 662 | 412 |
| 64 | **2,264** | 2,224 | 1,840 | 699 |
| 128 | 1,983 | 1,904 | 1,830 | 612 |
| 256 | 2,220 | 1,974 | 1,987 | 613 |

- **TP=2 is the best Sage setup.** DP=2 matched its throughput but requests hung on the real extraction workload
  (timeouts after 900 s). Two separate single-GPU servers did not load under WSL2 (`CUDA driver error: device not
  ready` while repacking the MoE weights, or no room left for the KV cache).
- **MTP speculative decoding is slower** (mean acceptance 2.28 of 3, still a net loss).
- **64 parallel requests** is the sweet spot; more only queues. The fp8 KV cache holds 1.77M tokens (27 full 64k
  requests at once), against ~140k for the dense model.

Serve script: [integrations/vllm/serve-sage-a1.5.sh](../integrations/vllm/serve-sage-a1.5.sh).

### Claim extraction (122 real source chunks, 64 in parallel)

| model | time | claims | quotes found in the source |
|---|---|---|---|
| Qwen3.8-27B | 340 s | 586 | 98.1% |
| Sage-A1.5, previous extraction prompt | 73 s | 397 | 98.2% |
| Sage-A1.5, current prompt | 76 s | 512 | 98.4% |

Sage returned no claims for many on-topic pages from adjacent fields: it read "skip anything unrelated" strictly.
The current prompt says that evidence from adjacent fields, datasets, markets or periods counts as related (the report
labels it as analogy). That recovered most of the gap, and it did not add claims from pages both models found
unrelated.

### Full runs: the same question, three ways

Question: *Why did the Ottoman siege of Vienna in 1683 fail?* (profile `normal`, Turkish report). The hybrid reused the
Sage run's collected evidence, and only the report phases ran again on the dense model.

| | dense 27B only | Sage only | **hybrid: Sage collects, 27B writes** |
|---|---|---|---|
| wall time | 5 h 02 min | **1 h 42 min** | 2 h 20 min (70 + ~6 swap + 64) |
| grounded claim clusters | 2,014 | **2,288** | 2,288 |
| high-confidence clusters | 183 | **340** | 340 |
| verified yield per minute (collect) | ~1.7 | **~9.1** | ~9.1 |
| report words | **43,354** | 35,650 | 38,501 |
| confidence tags in the report | 189 | 81 | **227** |
| "high" tags backed by 2+ sources | 33 of 34 | 18 of 22 | **47 of 47** |
| tags without a source | **0** | 9 | **0** |
| paragraphs without a citation | **12** | 30 | **12** |
| unexplained numbers | 4 | 7 | **0** |
| cited URLs (scholarly) | **205 (80)** | 156 (57) | 168 (64) |

The dense-only run is from an earlier day with 16 parallel requests and the previous extraction prompt. The Sage and
hybrid runs used 64 parallel requests and the current prompt.

Reading: the MoE model collects more verified evidence several times faster. The dense model writes the better report,
including from the MoE model's evidence. The hybrid has the best report checks in under half the dense model's time.
The dense-only report is still longer and cites more scholarly sources.

## Running the hybrid

Both models do not fit on 2x 24 GB at once, so ldr swaps them between the stages. Start from
[integrations/vllm/hybrid.ldr.toml](../integrations/vllm/hybrid.ldr.toml):

```toml
[llm]                      # the writer; also every setting a stage does not override
base_url = "http://127.0.0.1:8000/v1"
context = 65536
concurrency = 64
no_think = true

[collect]
base_url = "http://127.0.0.1:8002/v1"
model = "sage-a1.5"
start_cmd = "integrations/vllm/vllm-bg.sh start sage integrations/vllm/serve-sage-a1.5.sh"
stop_cmd = "integrations/vllm/vllm-bg.sh stop sage"

[report]
start_cmd = "integrations/vllm/vllm-bg.sh start writer vllm serve <dense model> --port 8000 ..."
stop_cmd = "integrations/vllm/vllm-bg.sh stop writer"
```

How the switch works:

- The switch happens when the first phase of a stage actually runs. Entering a stage runs the other stage's
  `stop_cmd`, then this stage's `start_cmd`, then waits until the server lists the model (`stage_ready_timeout`,
  default 1800 s).
- A resumed run whose collecting is finished never starts the collecting model.
- Commands run without a shell. Use a script for pipes or `&&`. `systemctl --user start ...` works too.
- The run log names each stage's model and server.
- `ldr doctor` and the pre-run check test each stage's server. A stage with a `start_cmd` is skipped there, because
  ldr starts that server itself.

Without `start_cmd` / `stop_cmd`, both servers must be running, for example on two machines, or a local collector
with a hosted writer.

## Hosted APIs

Any OpenAI-compatible endpoint works, for one stage or for both. Set `base_url`, `api_key` and `model`:

| provider | base_url |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Anthropic | `https://api.anthropic.com/v1/` (OpenAI-compatible endpoint) |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Mistral | `https://api.mistral.ai/v1` |

Notes:

- `token_param = "max_completion_tokens"` for models that refuse `max_tokens` (OpenAI reasoning models, GPT-5).
- `temperature = false` for models that accept only their default temperature.
- `json_mode = "none"` if the API rejects `response_format: json_object`. Answers are then parsed from plain text.
- Keep `no_think = false`: hosted APIs may reject the `chat_template_kwargs` field it adds.
- Set `concurrency` to what your rate limit allows. Throttled answers (HTTP 429, 5xx, 529) are resent after the
  server's `Retry-After` (else 5, 10, 20 ... s, at most 120 s).
- A hosted model sees the search results and page excerpts it is asked about. Use a local model for any stage whose
  material must stay on your machine.

Example: local collecting, hosted writing.

```toml
[llm]
base_url = "http://127.0.0.1:8002/v1"     # local MoE model for everything ...
no_think = true
concurrency = 64

[report]                                    # ... except the report
base_url = "https://api.openai.com/v1"
model = "gpt-5"
api_key = "sk-..."                          # or LDR_REPORT_API_KEY in the environment
no_think = false
token_param = "max_completion_tokens"
temperature = false
concurrency = 8
context = 200000
```
