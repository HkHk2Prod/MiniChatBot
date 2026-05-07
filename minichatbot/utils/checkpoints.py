"""Helpers for locating training runs and their checkpoints.

The naming convention `runs/{YYYYMMDD_HHMMSS}_{run_name}/checkpoints/ckpt_step_NNNNNNNN.pt`
means alphabetical sort = creation order, so finding "latest" is just `max(...)`.

Centralized here so future stages (SFT, RL) can bootstrap from a pretrain
checkpoint without each script reinventing the directory walk.
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_GLOB = "ckpt_step_*.pt"


def find_latest_run(
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path | None:
    """Return the most recent run directory under `output_dir`, or None.

    If `run_name` is given, only runs whose directory ends with `_{run_name}`
    are considered (e.g., `run_name="pretrain_tinystories"`).
    """
    out = Path(output_dir)
    if not out.exists():
        return None
    candidates = [p for p in out.iterdir() if p.is_dir()]
    if run_name is not None:
        candidates = [p for p in candidates if p.name.endswith(f"_{run_name}")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the most recent `ckpt_step_*.pt` within a run dir, or None."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob(CHECKPOINT_GLOB))
    return ckpts[-1] if ckpts else None


def find_latest_checkpoint_in(
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path | None:
    """Walk `output_dir`, pick the latest run, return its latest checkpoint."""
    run = find_latest_run(output_dir, run_name=run_name)
    if run is None:
        return None
    return find_latest_checkpoint(run)
