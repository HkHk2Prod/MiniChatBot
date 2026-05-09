"""Base class for batch collators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from minichatbot.tokenizer.base import Tokenizer


class Collator(ABC):
    """Assembles a list of samples into a batched tensor dict.

    Stage-specific subclasses define how raw dataset samples become
    model-ready batches: pretrain packs sequences, SFT pads with loss
    masks, RL preference pairs chosen and rejected sequences.
    """

    @abstractmethod
    def __call__(self, samples: list[Any]) -> dict[str, torch.Tensor]: ...

    @classmethod
    def from_config(cls, tokenizer: Tokenizer) -> Collator:
        """Default: no-arg construction. Override when the collator needs
        tokenizer state (e.g. SFT needs `pad_id`)."""
        return cls()
