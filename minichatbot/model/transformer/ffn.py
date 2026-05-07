"""Feed-forward network for the transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU FFN: out = W_down( silu(W_gate(x)) * W_up(x) ).

    Three matrices instead of GELU FFN's two, in exchange for higher
    expressivity at similar param budget. Standard in modern decoder-only
    LMs (LLaMA, Mistral, Qwen).
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
