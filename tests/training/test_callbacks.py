"""Tests for training callbacks (minichatbot/training/callbacks/*.py).

Covers the CheckpointCallback's pruning / best-tracking logic against a fake
trainer that just touches files, the console elapsed-time formatter, and the
LmEvalCallback's start/end phase gating.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from minichatbot.config import Config, DataConfig
from minichatbot.training.callbacks.base import CallbackContext
from minichatbot.training.callbacks.checkpoint import CheckpointCallback
from minichatbot.training.callbacks.console import _fmt_elapsed
from minichatbot.training.callbacks.lm_eval_callback import LmEvalCallback


class _FakeTrainer:
    """Stand-in for Trainer.save_checkpoint that just creates the file."""

    def save_checkpoint(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


def _ctx(run_dir: Path, step: int, eval_metrics: dict | None = None) -> CallbackContext:
    return CallbackContext(
        step=step,
        epoch=0,
        run_dir=run_dir,
        config=Config(run_name="t", data=DataConfig(train_path="x")),
        model=nn.Linear(2, 2),
        trainer=_FakeTrainer(),
        eval_metrics=eval_metrics,
    )


def _step_ckpts(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_step_*.pt"))


def test_keep_last_k_prunes_old_step_checkpoints(tmp_path: Path) -> None:
    cb = CheckpointCallback(every=1, keep_last_k=2)
    for step in (1, 2, 3, 4):
        cb.on_step_end(_ctx(tmp_path, step))
    assert _step_ckpts(tmp_path) == ["ckpt_step_00000003.pt", "ckpt_step_00000004.pt"]


def test_prunes_before_saving_to_cap_disk_peak(tmp_path: Path) -> None:
    # Prune-before-save means that at the instant a new checkpoint is written,
    # disk holds at most keep_last_k-1 *older* step checkpoints — so the peak
    # (older files + the one being written) stays at keep_last_k, never +1.
    seen_at_save: list[int] = []

    class _CountingTrainer:
        def save_checkpoint(self, path) -> None:
            d = Path(path).parent
            d.mkdir(parents=True, exist_ok=True)
            seen_at_save.append(len(list(d.glob("ckpt_step_*.pt"))))
            Path(path).touch()

    cb = CheckpointCallback(every=1, keep_last_k=2)
    for step in (1, 2, 3, 4, 5):
        ctx = _ctx(tmp_path, step)
        ctx.trainer = _CountingTrainer()
        cb.on_step_end(ctx)

    assert max(seen_at_save) <= 1  # keep_last_k - reserve(1); never keep_last_k+1
    assert _step_ckpts(tmp_path) == ["ckpt_step_00000004.pt", "ckpt_step_00000005.pt"]


def test_best_checkpoint_survives_pruning(tmp_path: Path) -> None:
    cb = CheckpointCallback(every=1, keep_last_k=1)
    cb.on_step_end(_ctx(tmp_path, 1))
    (tmp_path / "checkpoints" / "ckpt_best.pt").touch()
    cb.on_step_end(_ctx(tmp_path, 2))
    cb.on_step_end(_ctx(tmp_path, 3))
    assert (tmp_path / "checkpoints" / "ckpt_best.pt").exists()
    assert _step_ckpts(tmp_path) == ["ckpt_step_00000003.pt"]


def test_periodic_save_respects_every(tmp_path: Path) -> None:
    cb = CheckpointCallback(every=5, keep_last_k=None)
    cb.on_step_end(_ctx(tmp_path, 3))  # not a multiple of 5 -> no save
    assert _step_ckpts(tmp_path) == []
    cb.on_step_end(_ctx(tmp_path, 5))  # multiple of 5 -> save
    assert _step_ckpts(tmp_path) == ["ckpt_step_00000005.pt"]


def test_best_tracking_only_saves_on_improvement(tmp_path: Path) -> None:
    cb = CheckpointCallback()
    first = _ctx(tmp_path, 10, eval_metrics={"loss": 2.0})
    cb.on_eval_end(first)
    assert first.extra["best_step"] == 10 and first.extra["best_loss"] == 2.0

    worse = _ctx(tmp_path, 20, eval_metrics={"loss": 3.0})
    cb.on_eval_end(worse)
    assert "best_loss" not in worse.extra  # 3.0 did not improve on 2.0

    better = _ctx(tmp_path, 30, eval_metrics={"loss": 1.0})
    cb.on_eval_end(better)
    assert better.extra["best_loss"] == 1.0
    assert better.extra["best_step"] == 30


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00"),
        (65, "00:01:05"),
        (3661, "01:01:01"),
        (360000, "100:00:00"),  # hours are unbounded
    ],
)
def test_fmt_elapsed(seconds: int, expected: str) -> None:
    assert _fmt_elapsed(seconds) == expected


def test_lm_eval_rejects_unknown_eval_at() -> None:
    with pytest.raises(ValueError, match="eval_at"):
        LmEvalCallback(target_tasks=["piqa"], eval_at="sometimes")


@pytest.mark.parametrize(
    ("eval_at", "expected"),
    [
        ("end", ["end"]),  # default: only the result
        ("start", ["start"]),  # only the input baseline
        ("both", ["start", "end"]),  # baseline then result, in fire order
    ],
)
def test_lm_eval_phase_gating(tmp_path: Path, eval_at: str, expected: list[str]) -> None:
    # on_train_start / on_train_end should invoke the eval only for the phases
    # the eval_at setting selects — without actually importing lm_eval/torch.
    cb = LmEvalCallback(target_tasks=["piqa"], eval_at=eval_at)
    fired: list[str] = []
    cb._run = lambda ctx, phase="end": fired.append(phase)  # type: ignore[method-assign]
    ctx = _ctx(tmp_path, step=0)
    cb.on_train_start(ctx)
    cb.on_train_end(ctx)
    assert fired == expected
