"""Shared embedding helper for research tools.

The bigger multilingual model (config embed_model_gpu, bge-m3 by default) runs on a GPU only when that GPU really has
room; otherwise the small CPU model (config embed_model, all-MiniLM-L6-v2). bge-m3 on CPU is too slow (~1.7 chunks/s).
Device choice (config embed_device: "auto" | "cpu" | "cuda" | "cuda:N" | "mps"; auto by default):
  - nvidia-smi present: the GPU with the most free memory by nvidia-smi used/total, if > GPU_MIN_FREE_MB.
    WSL2 trap (2026-10-17): torch.cuda.mem_get_info reported 23.6 GB free while the LLM server held 24.1/24.6 GB,
    so bge-m3 went to the GPU, spilled into shared system memory and crawled. nvidia-smi is the trusted source.
  - no nvidia-smi (ROCm, a driver without the tool): torch.cuda.mem_get_info, but never under WSL (the trap above).
  - Apple silicon: MPS when the Metal working set has room. Unified memory is shared with a local LLM server, so the
    same free-memory floor applies.
  - anything missing or failing: CPU. Detection never raises.
"""
import os
import platform
import shutil
import subprocess

from . import config as _config

_S = _config.load()  # file + env only
GPU_MODEL, CPU_MODEL = _S.embed_model_gpu, _S.embed_model
DEVICE = str(getattr(_S, "embed_device", "auto") or "auto").strip().lower()
GPU_MIN_FREE_MB = 3500  # bge-m3 weights + activations at batch 32 with a margin
_cache = {}


def cpu_threads(cap=16):
    """torch CPU threads: the CPUs this process may use, at most `cap` (16 measured best on a 24-core box; more
    threads than cores only adds contention on a laptop)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(cap, n))


def _wsl():
    return "microsoft" in platform.release().lower()


def free_gpu():
    """(index, free MiB) of the emptiest NVIDIA GPU from nvidia-smi used/total, or None (no tool, no GPU, error)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10).stdout
        rows = [tuple(int(x) for x in line.split(",")) for line in out.strip().splitlines()]
        i, used, total = max(rows, key=lambda r: r[2] - r[1])
        return i, total - used
    except Exception:  # noqa: BLE001  no usable answer = no GPU
        return None


def _torch_gpu():
    """("cuda:N" | "mps", free MiB) from torch when nvidia-smi is absent, or None. Skipped under WSL (see above)."""
    try:
        import torch
    except Exception:  # noqa: BLE001  torch missing or broken: CPU
        return None
    try:
        if torch.cuda.is_available() and not _wsl():
            best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
            return f"cuda:{best}", torch.cuda.mem_get_info(best)[0] // 2 ** 20
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            room = torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
            return "mps", room // 2 ** 20
    except Exception:  # noqa: BLE001
        pass
    return None


def pick():
    """Model name + device actually used, so an index records what built it."""
    if DEVICE == "cpu" or not GPU_MODEL:
        return CPU_MODEL, "cpu"
    if DEVICE != "auto":  # an explicit device: the user knows it has room
        return GPU_MODEL, DEVICE
    g = free_gpu()
    if g is not None:
        return (GPU_MODEL, f"cuda:{g[0]}") if g[1] > GPU_MIN_FREE_MB else (CPU_MODEL, "cpu")
    t = _torch_gpu()
    if t is not None and t[1] > GPU_MIN_FREE_MB:
        return GPU_MODEL, t[0]
    return CPU_MODEL, "cpu"


def embed(texts, model=None, device=None):
    import torch
    from sentence_transformers import SentenceTransformer
    if model is None:
        model, device = pick()
    device = device or ("cpu" if model == CPU_MODEL else pick()[1])
    if device == "cpu":
        torch.set_num_threads(cpu_threads())
    key = (model, device)
    if key not in _cache:
        m = SentenceTransformer(model, device=device)
        m.max_seq_length = 512 if model == GPU_MODEL else 256
        _cache[key] = m
    return _cache[key].encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
