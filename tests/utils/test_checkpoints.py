"""Tests for run/checkpoint discovery (minichatbot/utils/checkpoints.py).

Pure filesystem logic — exercised with empty placeholder files, no torch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minichatbot.config import Config, DataConfig
from minichatbot.utils.checkpoints import (
    find_best_checkpoint,
    find_latest_checkpoint,
    find_latest_checkpoint_in,
    find_latest_run,
    resolve_pretrained_arg,
    resolve_resume_arg,
)


def _make_run(root: Path, name: str, steps: list[int] = (), *, best: bool = False) -> Path:
    """Create runs/<name>/checkpoints/ with placeholder ckpt files."""
    ckpt_dir = root / name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for s in steps:
        (ckpt_dir / f"ckpt_step_{s:08d}.pt").touch()
    if best:
        (ckpt_dir / "ckpt_best.pt").touch()
    return root / name


def _config(output_dir: Path, run_name: str = "demo") -> Config:
    return Config(run_name=run_name, data=DataConfig(train_path="x"), output_dir=str(output_dir))


def test_find_latest_run_picks_newest_by_name(tmp_path: Path) -> None:
    _make_run(tmp_path, "20260101_000000_demo")
    newest = _make_run(tmp_path, "20260507_120000_demo")
    assert find_latest_run(tmp_path) == newest


def test_find_latest_run_filters_by_run_name(tmp_path: Path) -> None:
    other = _make_run(tmp_path, "20260601_000000_other")  # newer but different name
    demo = _make_run(tmp_path, "20260101_000000_demo")
    assert find_latest_run(tmp_path, run_name="demo") == demo
    assert find_latest_run(tmp_path) == other  # unfiltered picks the newest overall


def test_find_latest_run_missing_dir_returns_none(tmp_path: Path) -> None:
    assert find_latest_run(tmp_path / "nope") is None


def test_find_latest_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "20260101_000000_demo", steps=[100, 200, 1000])
    assert find_latest_checkpoint(run).name == "ckpt_step_00001000.pt"


def test_find_latest_checkpoint_none_when_empty(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "20260101_000000_demo")
    assert find_latest_checkpoint(run) is None


def test_find_best_checkpoint(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "20260101_000000_demo", steps=[100], best=True)
    assert find_best_checkpoint(run).name == "ckpt_best.pt"
    no_best = _make_run(tmp_path, "20260102_000000_demo", steps=[100])
    assert find_best_checkpoint(no_best) is None


def test_find_latest_checkpoint_in_falls_through_empty_runs(tmp_path: Path) -> None:
    # Newest run has no checkpoints; discovery should fall through to the older one.
    _make_run(tmp_path, "20260507_000000_demo")  # newest, empty
    older = _make_run(tmp_path, "20260101_000000_demo", steps=[100])
    found = find_latest_checkpoint_in(tmp_path, run_name="demo")
    assert found == older / "checkpoints" / "ckpt_step_00000100.pt"


def test_resolve_resume_auto(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "20260101_000000_demo", steps=[100, 200])
    cfg = _config(tmp_path)
    assert resolve_resume_arg("auto", cfg) == run / "checkpoints" / "ckpt_step_00000200.pt"


def test_resolve_resume_explicit_file_and_dir(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "20260101_000000_demo", steps=[300])
    cfg = _config(tmp_path)
    ckpt = run / "checkpoints" / "ckpt_step_00000300.pt"
    assert resolve_resume_arg(str(ckpt), cfg) == ckpt  # explicit file
    assert resolve_resume_arg(str(run), cfg) == ckpt  # dir -> latest within


def test_resolve_resume_missing_raises(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with pytest.raises(SystemExit):
        resolve_resume_arg(str(tmp_path / "does_not_exist"), cfg)


def test_resolve_pretrained_prefers_best(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "20260101_000000_demo", steps=[100, 200], best=True)
    cfg = _config(tmp_path)
    assert resolve_pretrained_arg(str(run), cfg) == run / "checkpoints" / "ckpt_best.pt"
