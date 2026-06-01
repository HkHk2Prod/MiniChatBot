"""Heuristic wall-clock estimate for a training run.

Uses the Kaplan compute approximation
    C ≈ 6 N D    (forward + backward FLOPs per token ≈ 6 × params)
and divides by the GPU's peak bf16 tensor-core throughput times a
fixed model-FLOPs-utilization (MFU) factor. Activation checkpointing
bumps the multiplier to 7 N D (extra forward pass during backward).

Off-table GPUs and non-CUDA devices return an "unknown" string rather
than fabricate a number.
"""

from __future__ import annotations

import torch

# Peak bf16 (tensor-core, dense, no sparsity) TFLOPS for common NVIDIA
# training GPUs — from each card's datasheet.
_GPU_BF16_TFLOPS: dict[str, float] = {
    "H100": 989.0,
    "A100": 312.0,
    "L40": 181.0,
    "L4": 121.0,
    "V100": 125.0,  # fp16 (no native bf16); used as a proxy
    "T4": 65.0,
    "RTX 4090": 165.0,
    "RTX 4080": 97.0,
    "RTX 3090": 71.0,
    "RTX A6000": 154.0,
}

_DEFAULT_MFU = 0.30
# Tensor-core fp32 (non-TF32) is ~1/8 of bf16 peak on Ampere/Hopper.
_FP32_DERATE = 1.0 / 8.0


def peak_tflops_for(device_name: str) -> float | None:
    """Return peak bf16 TFLOPS for `device_name`, or None if not tabulated."""
    for key, tflops in _GPU_BF16_TFLOPS.items():
        if key in device_name:
            return tflops
    return None


def _precision_multiplier(precision: str) -> float:
    return 1.0 if precision in ("bf16", "fp16") else _FP32_DERATE


def estimate_runtime_seconds(
    n_params: int,
    tokens_seen: int,
    peak_tflops: float,
    precision: str = "bf16",
    grad_checkpointing: bool = False,
    mfu: float = _DEFAULT_MFU,
) -> float:
    flops_per_token = 7.0 if grad_checkpointing else 6.0
    total = flops_per_token * n_params * tokens_seen
    effective = peak_tflops * 1e12 * _precision_multiplier(precision) * mfu
    return total / effective


def _fmt_hm(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"~{d}d{h:02d}h"
    return f"~{h}h{m:02d}m"


def format_runtime_estimate(
    n_params: int,
    tokens_seen: int,
    device: torch.device,
    precision: str = "bf16",
    grad_checkpointing: bool = False,
    mfu: float = _DEFAULT_MFU,
) -> str:
    """One-line estimate suitable for the run-start banner.

    Returns an "unknown (reason)" string when the device is non-CUDA or
    the GPU isn't in the perf table — better than a fabricated number.
    """
    if device.type != "cuda":
        return f"unknown ({device.type} device, needs CUDA for perf table)"
    name = torch.cuda.get_device_name(device)
    tflops = peak_tflops_for(name)
    if tflops is None:
        return f"unknown ({name!r} not in perf table)"
    secs = estimate_runtime_seconds(
        n_params, tokens_seen, tflops, precision, grad_checkpointing, mfu
    )
    return f"{_fmt_hm(secs)} (est. @ {mfu:.0%} MFU on {name}, {tflops:.0f} TFLOPS bf16)"
