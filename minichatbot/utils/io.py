"""I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(state: Any, path: str | Path) -> None:
    """torch.save with an atomic write.

    Writes to `{path}.tmp` first, then renames. A crash during torch.save
    leaves the .tmp file behind but never a half-written `path` — readers
    of `path` always see either the previous contents or the new ones.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(p)
