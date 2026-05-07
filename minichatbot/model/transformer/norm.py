"""Normalization layers used in the transformer."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (LLaMA-style).

    Skips the mean-subtraction step of LayerNorm. Slightly cheaper and
    used by most modern decoder-only LMs. Computes in fp32 internally
    to stay stable under bf16/fp16 autocast, then casts back.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(dtype) * self.weight


def make_norm(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")
