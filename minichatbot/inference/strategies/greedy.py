"""Greedy (argmax) sampling strategy."""

from __future__ import annotations

import torch

from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.inference.strategies.base import SamplingStrategy


@SAMPLING_REGISTRY.register("greedy")
class GreedySampling(SamplingStrategy):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)
