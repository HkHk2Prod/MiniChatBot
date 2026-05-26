"""Tests for optimizer / LR-scheduler builders (minichatbot/training/optim.py)."""

from __future__ import annotations

import math

import pytest

from minichatbot.config import OptimConfig
from minichatbot.training.optim import (
    _constant_lambda,
    _cosine_lambda,
    _linear_lambda,
    _split_param_groups,
    build_optimizer,
    build_scheduler,
)


def test_warmup_is_linear_ramp() -> None:
    fn = _cosine_lambda(warmup=10, total=100, min_ratio=0.1)
    assert fn(0) == pytest.approx(1 / 10)   # step+1 over warmup
    assert fn(4) == pytest.approx(5 / 10)
    assert fn(9) == pytest.approx(10 / 10)  # last warmup step reaches full LR


def test_cosine_endpoints_and_midpoint() -> None:
    fn = _cosine_lambda(warmup=0, total=100, min_ratio=0.1)
    assert fn(0) == pytest.approx(1.0)            # progress 0 -> full
    assert fn(100) == pytest.approx(0.1)          # progress 1 -> floor
    assert fn(50) == pytest.approx((1 + 0.1) / 2)  # progress 0.5 -> mean of 1 and floor


def test_cosine_clamps_past_total() -> None:
    fn = _cosine_lambda(warmup=0, total=100, min_ratio=0.1)
    assert fn(250) == pytest.approx(0.1)  # progress clamped to 1.0


def test_linear_endpoints() -> None:
    fn = _linear_lambda(warmup=0, total=100, min_ratio=0.2)
    assert fn(0) == pytest.approx(1.0)
    assert fn(50) == pytest.approx(1.0 - 0.8 * 0.5)
    assert fn(100) == pytest.approx(0.2)
    assert fn(999) == pytest.approx(0.2)  # floored at min_ratio


def test_constant_after_warmup() -> None:
    fn = _constant_lambda(warmup=5)
    assert fn(0) == pytest.approx(1 / 5)
    assert fn(5) == pytest.approx(1.0)
    assert fn(1000) == pytest.approx(1.0)


def test_split_param_groups_decay_vs_no_decay(tiny_transformer) -> None:
    decay, no_decay = _split_param_groups(tiny_transformer, weight_decay=0.1)
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # Matrices/embeddings (ndim >= 2) decay; biases/norm gains (ndim < 2) don't.
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])
    assert no_decay["params"], "norm gains should populate the no-decay group"


def test_build_scheduler_applies_lambda(tiny_transformer) -> None:
    cfg = OptimConfig(lr=0.01, warmup_steps=0, lr_schedule="cosine", min_lr_ratio=0.1)
    opt = build_optimizer(tiny_transformer, cfg)
    sched = build_scheduler(opt, cfg, max_steps=100)
    # LambdaLR sets lr = base_lr * fn(step); at construction step==0 -> full lr.
    assert sched.get_last_lr()[0] == pytest.approx(0.01)
    for _ in range(50):
        opt.step()
        sched.step()
    expected = 0.01 * (0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(math.pi * 0.5)))
    assert sched.get_last_lr()[0] == pytest.approx(expected)


def test_build_scheduler_unknown_raises(tiny_transformer) -> None:
    cfg = OptimConfig(lr_schedule="exponential")
    opt = build_optimizer(tiny_transformer, cfg)
    with pytest.raises(ValueError, match="Unknown lr_schedule"):
        build_scheduler(opt, cfg, max_steps=10)
