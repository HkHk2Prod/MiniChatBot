"""SFT loss: cross-entropy with -100 ignored (assistant-only masking).

The loss math is identical to PretrainLoss — what makes SFT different
is the *data*, not the loss function. The SFTDataset emits `labels`
already shifted-and-masked: -100 for system/user/role-tag tokens,
real ids for assistant content. This loss just defers to PretrainLoss.

Registered under "sft" so configs can say `--loss sft` for clarity.
"""

from __future__ import annotations

from minichatbot.training.losses import LOSS_REGISTRY
from minichatbot.training.losses.pretrain import PretrainLoss


@LOSS_REGISTRY.register("sft")
class SFTLoss(PretrainLoss):
    """Alias of PretrainLoss for naming clarity in SFT configs."""
