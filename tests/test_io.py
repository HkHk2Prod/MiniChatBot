"""Tests for atomic checkpoint writes (minichatbot/utils/io.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_removes_partial_tmp_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed write (e.g. ENOSPC) must not strand its partial .tmp."""
    path = tmp_path / "ckpt.pt"

    def boom(state: object, f: object, *args: object, **kwargs: object) -> None:
        Path(f).write_bytes(b"partial bytes")  # simulate a partial .tmp on disk
        raise RuntimeError("file write failed")  # ...then the write blows up

    monkeypatch.setattr("minichatbot.utils.io.torch.save", boom)
    with pytest.raises(RuntimeError, match="file write failed"):
        atomic_torch_save({"x": 1}, path)
    assert not path.exists()                # the destination was never created
    assert list(tmp_path.iterdir()) == []   # and the partial .tmp was cleaned up


def test_failed_overwrite_keeps_previous_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed re-save leaves the prior `path` untouched (true atomicity)."""
    path = tmp_path / "ckpt.pt"
    atomic_torch_save({"step": 1}, path)  # establish a good checkpoint

    def boom(state: object, f: object, *args: object, **kwargs: object) -> None:
        Path(f).write_bytes(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr("minichatbot.utils.io.torch.save", boom)
    with pytest.raises(RuntimeError):
        atomic_torch_save({"step": 2}, path)
    assert torch.load(path, weights_only=False)["step"] == 1  # old contents survive
    assert list(tmp_path.iterdir()) == [path]                 # no orphaned .tmp
