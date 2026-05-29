"""Tests for transformer building blocks: normalization and RoPE attention.

(minichatbot/model/transformer/{norm,attention}.py)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from minichatbot.model.transformer.attention import (
    KVCache,
    MultiHeadAttention,
    apply_rope,
    precompute_rope_cache,
)
from minichatbot.model.transformer.norm import RMSNorm, make_norm


def test_rmsnorm_matches_manual() -> None:
    norm = RMSNorm(8)  # weight initialized to ones
    x = torch.randn(2, 4, 8)
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps)
    assert torch.allclose(norm(x), x * rms, atol=1e-6)


def test_rmsnorm_applies_weight() -> None:
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.randn(1, 3, 4)
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps)
    assert torch.allclose(norm(x), (x * rms) * norm.weight, atol=1e-6)


def test_make_norm_dispatch() -> None:
    assert isinstance(make_norm("rmsnorm", 8), RMSNorm)
    assert isinstance(make_norm("layernorm", 8), nn.LayerNorm)
    with pytest.raises(ValueError, match="Unknown norm_type"):
        make_norm("groupnorm", 8)


def test_rope_cache_shape() -> None:
    cos, sin = precompute_rope_cache(seq_len=16, head_dim=8)
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)


def test_rope_cache_requires_even_head_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        precompute_rope_cache(seq_len=16, head_dim=7)


def test_apply_rope_preserves_norm() -> None:
    # RoPE is a rotation, so it preserves each vector's L2 norm.
    cos, sin = precompute_rope_cache(seq_len=5, head_dim=8)
    x = torch.randn(1, 2, 5, 8)  # (batch, heads, T, head_dim)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_attention_cache_grows_with_each_step() -> None:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    mha.eval()
    cos, sin = precompute_rope_cache(16, 4)
    cache = KVCache(torch.empty(1, 2, 0, 4), torch.empty(1, 2, 0, 4))
    for expected_len in (1, 2, 3):
        x = torch.randn(1, 1, 8)
        _, cache = mha(x, cos, sin, cache=cache)
        assert cache.k.size(2) == expected_len


def test_attention_chunked_decode_matches_full_forward() -> None:
    # Exercises the explicit bottom-right causal-mask branch (cache present,
    # q_len > 1) and checks it agrees with a single is_causal forward.
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    mha.eval()
    cos, sin = precompute_rope_cache(16, 4)
    x = torch.randn(1, 6, 8)
    with torch.no_grad():
        full, _ = mha(x, cos, sin, cache=None)
        empty = KVCache(torch.empty(1, 2, 0, 4), torch.empty(1, 2, 0, 4))
        _, c1 = mha(x[:, :2], cos, sin, cache=empty)  # prefill 2
        chunk, _ = mha(x[:, 2:], cos, sin, cache=c1)  # chunked decode of 4
    assert torch.allclose(chunk, full[:, 2:], atol=1e-5)
