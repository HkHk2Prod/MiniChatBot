"""Collator for pretraining: stacks (seq_len + 1) chunks and shifts."""

from __future__ import annotations

import torch

from minichatbot.data.collators import COLLATOR_REGISTRY
from minichatbot.data.collators.base import Collator


@COLLATOR_REGISTRY.register("pretrain")
class PretrainCollator(Collator):
    """Stacks (seq_len + 1) chunks and produces input_ids / labels.

    input_ids = chunk[:-1]   (B, seq_len)
    labels    = chunk[1:]    (B, seq_len)
    """

    def __call__(self, samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = torch.stack(samples, dim=0)
        return {
            "input_ids": batch[:, :-1].contiguous(),
            "labels": batch[:, 1:].contiguous(),
        }
