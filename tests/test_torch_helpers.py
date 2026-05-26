"""Tests for small torch helpers (minichatbot/utils/torch_helpers.py)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from minichatbot.utils.torch_helpers import eval_mode, resolve_device, unwrap_compiled


def test_resolve_device_explicit_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_available_device() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_eval_mode_toggles_and_restores_train() -> None:
    model = nn.Linear(2, 2)
    model.train()
    with eval_mode(model):
        assert not model.training
    assert model.training  # restored to train


def test_eval_mode_restores_on_exception() -> None:
    model = nn.Linear(2, 2)
    model.train()
    with pytest.raises(RuntimeError):  # noqa: SIM117 - want the inner with for clarity
        with eval_mode(model):
            raise RuntimeError("boom")
    assert model.training  # train mode restored despite the exception


def test_eval_mode_leaves_eval_model_in_eval() -> None:
    model = nn.Linear(2, 2)
    model.eval()
    with eval_mode(model):
        assert not model.training
    assert not model.training  # was eval before -> stays eval


def test_unwrap_compiled_passthrough_and_unwrap() -> None:
    model = nn.Linear(2, 2)
    assert unwrap_compiled(model) is model
    # torch.compile wraps the model and exposes it as `_orig_mod`.
    wrapper = types.SimpleNamespace(_orig_mod=model)
    assert unwrap_compiled(wrapper) is model
