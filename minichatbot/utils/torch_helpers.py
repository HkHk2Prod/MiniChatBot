"""Small torch helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """Return the original module if `model` was wrapped by `torch.compile`.

    `torch.compile` returns an `OptimizedModule` that registers the wrapped
    model as `_orig_mod`. Its `state_dict()` therefore emits `_orig_mod.*`
    keys, which won't `load_state_dict` into a fresh (uncompiled) instance —
    so every save / load path must go through the unwrapped module.
    """
    return getattr(model, "_orig_mod", model)


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
