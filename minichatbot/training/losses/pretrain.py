"""Cross-entropy loss for next-token pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minichatbot.model.base import ModelOutput
from minichatbot.training.losses import LOSS_REGISTRY
from minichatbot.training.losses.base import Loss


@LOSS_REGISTRY.register("pretrain")
class PretrainLoss(Loss):
    """Token-level cross-entropy.

    Expects `output.logits` of shape (B, T, V) and `batch['labels']` of
    shape (B, T). The collator already produced shifted labels, so this
    is a flat CE over all positions.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        output: ModelOutput,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = output.logits
        labels = batch["labels"]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
