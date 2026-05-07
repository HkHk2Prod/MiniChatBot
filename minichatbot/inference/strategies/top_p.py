"""Top-p (nucleus) sampling."""

from __future__ import annotations

import torch

from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.inference.strategies.base import SamplingStrategy


@SAMPLING_REGISTRY.register("top_p")
class TopPSampling(SamplingStrategy):
    """Sample from the smallest set of tokens whose cumulative prob >= p.

    Uses the HF convention: keeps a token whenever the cumulative mass
    *before* it is still <= p, so the boundary token is included.
    """

    def __init__(self, p: float = 0.9, temperature: float = 1.0) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        mask = (cumprobs - sorted_probs) > self.p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)         # (B, 1)
        return sorted_idx.gather(-1, choice).squeeze(-1)         # (B,)
