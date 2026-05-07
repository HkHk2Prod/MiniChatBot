"""Small torch helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval_mode(model: nn.Module) -> Iterator[None]:
    """Temporarily set `model.eval()`; restore train mode on exit even if an exception fires.

    Use whenever you need to run a non-training forward pass (eval, generation,
    sampling) on a model that may currently be in train mode. Without this, an
    exception during the eval/generation block leaves the model in eval mode,
    silently breaking subsequent training (dropout off, BatchNorm using running
    stats, etc.).
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()
