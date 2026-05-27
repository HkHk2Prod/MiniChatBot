"""I/O utilities."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(state: Any, path: str | Path) -> None:
    """torch.save with an atomic write.

    Writes to `{path}.tmp` first, then renames it over `path` — readers of
    `path` always see either the previous contents or the new ones, never a
    half-written file. If the write fails (e.g. ENOSPC on a full disk) or is
    interrupted, the partial `.tmp` is removed before the error propagates,
    so a failed save doesn't strand a multi-hundred-MB partial that would
    only make the next attempt's disk situation worse.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        torch.save(state, tmp)
        tmp.replace(p)
    except BaseException:  # noqa: BLE001 — clean up the partial, then re-raise as-is
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
