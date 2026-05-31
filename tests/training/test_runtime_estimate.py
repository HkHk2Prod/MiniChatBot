"""Tests for the run-start runtime estimate helper."""

from __future__ import annotations

import math

import pytest
import torch

from minichatbot.training import runtime_estimate as rte
from minichatbot.training.runtime_estimate import (
    _fmt_hm,
    estimate_runtime_seconds,
    format_runtime_estimate,
    peak_tflops_for,
)


def test_peak_tflops_for_known_substring() -> None:
    # Real device names include vendor + variant; substring match should hit.
    assert peak_tflops_for("NVIDIA A100-SXM4-40GB") == 312.0
    assert peak_tflops_for("NVIDIA H100 80GB HBM3") == 989.0
    assert peak_tflops_for("NVIDIA GeForce RTX 4090") == 165.0


def test_peak_tflops_for_unknown_returns_none() -> None:
    assert peak_tflops_for("NVIDIA RTX 9999") is None
    assert peak_tflops_for("") is None


def test_estimate_seconds_matches_6nd_formula() -> None:
    # 1B params, 1B tokens, 100 TFLOPS, 50% MFU → 6e18 / 5e13 = 1.2e5 s.
    secs = estimate_runtime_seconds(
        n_params=1_000_000_000,
        tokens_seen=1_000_000_000,
        peak_tflops=100.0,
        precision="bf16",
        mfu=0.5,
    )
    assert math.isclose(secs, 6.0e18 / 5.0e13, rel_tol=1e-6)


def test_estimate_seconds_bumps_with_grad_checkpointing() -> None:
    base = estimate_runtime_seconds(1_000_000, 1_000_000, 100.0)
    bumped = estimate_runtime_seconds(1_000_000, 1_000_000, 100.0, grad_checkpointing=True)
    # 7/6 bump from the extra forward pass during backward.
    assert math.isclose(bumped / base, 7.0 / 6.0, rel_tol=1e-6)


def test_estimate_seconds_fp32_is_derated() -> None:
    bf16 = estimate_runtime_seconds(1_000_000, 1_000_000, 100.0, precision="bf16")
    fp32 = estimate_runtime_seconds(1_000_000, 1_000_000, 100.0, precision="fp32")
    # fp32 tensor-core peak is ~1/8 of bf16, so fp32 should take ~8x longer.
    assert math.isclose(fp32 / bf16, 8.0, rel_tol=1e-6)


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "~0h00m"),
        (61, "~0h01m"),
        (3 * 3600 + 45 * 60, "~3h45m"),
        (2 * 86400 + 5 * 3600, "~2d05h"),
        (-1.0, "~0h00m"),
    ],
)
def test_fmt_hm(seconds: float, expected: str) -> None:
    assert _fmt_hm(seconds) == expected


def test_format_returns_unknown_for_non_cuda() -> None:
    msg = format_runtime_estimate(1_000_000, 1_000_000, torch.device("cpu"))
    assert msg.startswith("unknown")
    assert "cpu" in msg


def test_format_returns_unknown_for_off_table_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d: "NVIDIA RTX 9999")
    msg = format_runtime_estimate(1_000_000, 1_000_000, torch.device("cuda:0"))
    assert msg.startswith("unknown")
    assert "RTX 9999" in msg


def test_format_uses_table_for_known_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d: "NVIDIA A100-SXM4-40GB")
    # Pick numbers that produce a tidy result for visual inspection.
    msg = format_runtime_estimate(
        n_params=100_000_000,
        tokens_seen=10_000_000_000,
        device=torch.device("cuda:0"),
        precision="bf16",
        grad_checkpointing=False,
        mfu=0.30,
    )
    assert "A100" in msg
    assert "312 TFLOPS" in msg
    assert "30% MFU" in msg
    # The hour/minute prefix should look like "~Xh..m".
    assert msg.startswith("~")


def test_default_mfu_is_used_when_omitted(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d: "NVIDIA A100")
    msg = format_runtime_estimate(1_000_000, 1_000_000, torch.device("cuda:0"))
    assert f"{rte._DEFAULT_MFU:.0%} MFU" in msg
