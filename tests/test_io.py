"""Tests for atomic checkpoint writes (minichatbot/utils/io.py)."""

from __future__ import annotations

from pathlib import Path

import torch

from minichatbot.utils.io import atomic_torch_save


def test_round_trips_state(tmp_path: Path) -> None:
    state = {"step": 7, "tensor": torch.arange(4)}
    path = tmp_path / "ckpt.pt"
    atomic_torch_save(state, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["step"] == 7
    assert torch.equal(loaded["tensor"], torch.arange(4))


def test_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "ckpt.pt"
    atomic_torch_save({"x": 1}, path)
    assert path.exists()


def test_leaves_no_tmp_file_on_success(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    atomic_torch_save({"x": 1}, path)
    assert list(tmp_path.iterdir()) == [path]  # the .tmp was renamed away
