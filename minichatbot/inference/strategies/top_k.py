"""Top-k sampling."""

from __future__ import annotations

import torch

from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.inference.strategies.base import SamplingStrategy


@SAMPLING_REGISTRY.register("top_k")
class TopKSampling(SamplingStrategy):
    """Sample from the top-k logits (with optional temperature)."""

    def __init__(self, k: int, temperature: float = 1.0) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        k = min(self.k, logits.size(-1))
        topk_vals, topk_idx = logits.topk(k, dim=-1)
        probs = torch.softmax(topk_vals / self.temperature, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)        # (B, 1)
        return topk_idx.gather(-1, choice).squeeze(-1)          # (B,)
