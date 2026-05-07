"""Temperature-scaled sampling."""

from __future__ import annotations

import torch

from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.inference.strategies.base import SamplingStrategy


@SAMPLING_REGISTRY.register("temperature")
class TemperatureSampling(SamplingStrategy):
    """Sample from softmax(logits / temperature). temperature=0 → argmax."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
