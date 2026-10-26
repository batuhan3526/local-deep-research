#!/usr/bin/env bash
# Sage-A1.5 (InternScience, Qwen3.5-MoE 35B total / 3B active, hybrid linear + full attention), 4-bit W4A16 AutoRound,
# served by vLLM for the "collect" stage (thousands of short extraction / verification calls).
# Measured on 2x RTX 3090 (vLLM 0.28, power limit 55%): TP=2 beat data parallel and two single-GPU servers, MTP
# speculative decoding was slower; ~2,260 tok/s at 64 parallel requests, 1.77M tokens of fp8 KV cache. docs/models.md
# Overrides: MODEL PORT HOST TP MAXLEN UTIL SEQS GPUS
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
exec vllm serve "${MODEL:-stuchapin/Sage-A1.5-W4A16-AutoRound}" \
  --served-model-name sage-a1.5 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8002}" \
  --tensor-parallel-size "${TP:-2}" \
  --language-model-only \
  --gpu-memory-utilization "${UTIL:-0.90}" \
  --max-model-len "${MAXLEN:-65536}" \
  --max-num-seqs "${SEQS:-128}" \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching --mamba-cache-mode align \
  --async-scheduling \
  --reasoning-parser qwen3
