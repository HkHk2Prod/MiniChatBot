"""Base class for sampling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SamplingStrategy(ABC):
    """Picks next-token ids from a logits distribution.

    Encapsulates the full logits -> token-id pipeline. Concrete
    strategies (Greedy, Temperature, TopK, TopP) each implement one
    transformation. Composition (e.g., temperature -> top-k -> sample)
    lives in the Generator's loop, not at this interface.
    """

    @abstractmethod
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, V) -> next_token_ids: (B,)"""
