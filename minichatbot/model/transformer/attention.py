"""Multi-head self-attention with rotary position embeddings (RoPE).

Uses `torch.nn.functional.scaled_dot_product_attention` so Flash Attention
kernels are picked automatically when supported (Ampere+ GPU, contiguous
tensors, fp16/bf16).

Masking has three cases because `is_causal=True` is undefined behavior
in SDPA when q_len != k_len:
  1. Empty cache (training or first inference call): `is_causal=True`.
  2. Single-token decode (cache present, q_len == 1): no mask — the lone
     query attends to all cached keys plus itself.
  3. Chunked decode (cache present, q_len > 1): an explicit
     bottom-right-aligned causal mask of shape (q_len, kv_len).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache(NamedTuple):
    """Per-layer KV cache for autoregressive decoding."""
    k: torch.Tensor
    v: torch.Tensor


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape (seq_len, head_dim) in fp32."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                          # (seq_len, half)
    cos = freqs.cos().repeat_interleave(2, dim=-1)            # (seq_len, head_dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(a, b, c, d, ...) -> (-b, a, -d, c, ...) along the last dim."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to x.

    x:        (..., T, head_dim)
    cos, sin: (T, head_dim) — already sliced to the current positions.
    """
    return x * cos + _rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        pos_start = cache.k.size(2) if cache is not None else 0
        cos_slice = cos[pos_start : pos_start + T]
        sin_slice = sin[pos_start : pos_start + T]
        q = apply_rope(q, cos_slice, sin_slice)
        k = apply_rope(k, cos_slice, sin_slice)

        if cache is not None:
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)

        new_cache = KVCache(k=k, v=v) if cache is not None else None

        dropout_p = self.dropout if self.training else 0.0
        if pos_start == 0:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        elif T == 1:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            attn_mask = torch.ones(
                T, pos_start + T, dtype=torch.bool, device=q.device
            ).tril(diagonal=pos_start)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, attn_mask=attn_mask
            )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(out), new_cache
