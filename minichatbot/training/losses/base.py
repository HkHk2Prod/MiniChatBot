"""Base class for training losses."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from minichatbot.model.base import ModelOutput


class Loss(nn.Module, ABC):
    """Computes a scalar training loss from model output and a batch.

    Subclasses cover stage-specific objectives: cross-entropy for
    pretrain, masked cross-entropy for SFT, log-ratio losses for DPO,
    and policy losses for PPO. Inherits nn.Module so device placement
    and any internal parameters (e.g., learnable temperature) work
    uniformly.
    """

    @abstractmethod
    def forward(
        self,
        output: ModelOutput,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor: ...
