"""Single decoder-only transformer block (pre-norm, attention + FFN)."""

from __future__ import annotations

import torch
import torch.nn as nn

from minichatbot.model.transformer.attention import KVCache, MultiHeadAttention
from minichatbot.model.transformer.ffn import SwiGLU
from minichatbot.model.transformer.norm import make_norm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = make_norm(norm_type, d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias
        )
        self.ffn_norm = make_norm(norm_type, d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache
