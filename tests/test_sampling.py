"""Tests for sampling strategies (minichatbot/inference/strategies/*.py)."""

from __future__ import annotations

import pytest
import torch

from minichatbot.inference.strategies.greedy import GreedySampling
from minichatbot.inference.strategies.temperature import TemperatureSampling
from minichatbot.inference.strategies.top_k import TopKSampling
from minichatbot.inference.strategies.top_p import TopPSampling


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_greedy_equals_argmax() -> None:
    logits = torch.randn(4, 16)
    out = GreedySampling()(logits)
    assert torch.equal(out, logits.argmax(dim=-1))
    assert out.shape == (4,)


def test_temperature_zero_falls_back_to_greedy() -> None:
    logits = torch.randn(4, 16)
    out = TemperatureSampling(temperature=0.0)(logits)
    assert torch.equal(out, logits.argmax(dim=-1))


@pytest.mark.parametrize(
    "strategy",
    [
        GreedySampling(),
        TemperatureSampling(temperature=1.0),
        TopKSampling(k=5),
        TopPSampling(p=0.9),
    ],
)
def test_output_shape_is_batch(strategy: object) -> None:
    logits = torch.randn(3, 16)
    out = strategy(logits)  # type: ignore[operator]
    assert out.shape == (3,)
    assert out.dtype == torch.long
    assert torch.all((out >= 0) & (out < 16))


def test_top_k_never_samples_outside_top_k() -> None:
    # Two clearly-leading tokens; the bottom two must never be sampled.
    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0]]).repeat(64, 1)
    strategy = TopKSampling(k=2)
    samples = {int(strategy(logits)[i]) for _ in range(20) for i in range(64)}
    assert samples <= {0, 1}


def test_top_k_one_equals_argmax() -> None:
    logits = torch.randn(8, 16)
    out = TopKSampling(k=1)(logits)
    assert torch.equal(out, logits.argmax(dim=-1))


def test_top_p_excludes_negligible_mass() -> None:
    # Tokens 2 and 3 carry ~0 probability mass and must be filtered out.
    logits = torch.tensor([[5.0, 4.0, -50.0, -50.0]]).repeat(64, 1)
    strategy = TopPSampling(p=0.9)
    samples = {int(strategy(logits)[i]) for _ in range(20) for i in range(64)}
    assert samples <= {0, 1}


def test_top_p_tiny_p_keeps_only_top_token() -> None:
    logits = torch.randn(8, 16)
    out = TopPSampling(p=1e-6)(logits)
    assert torch.equal(out, logits.argmax(dim=-1))


def test_temperature_rejects_negative() -> None:
    with pytest.raises(ValueError, match="temperature"):
        TemperatureSampling(temperature=-0.5)


def test_top_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        TopKSampling(k=0)


@pytest.mark.parametrize("bad_p", [0.0, -0.1, 1.5])
def test_top_p_rejects_out_of_range_p(bad_p: float) -> None:
    with pytest.raises(ValueError, match="p must be"):
        TopPSampling(p=bad_p)
