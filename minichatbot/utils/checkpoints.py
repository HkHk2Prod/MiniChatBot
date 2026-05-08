"""Helpers for locating training runs and their checkpoints.

The naming convention `runs/{YYYYMMDD_HHMMSS}_{run_name}/checkpoints/ckpt_step_NNNNNNNN.pt`
means alphabetical sort = creation order, so finding "latest" is just `max(...)`.

A "best" checkpoint (lowest val loss seen during training) is saved as
`ckpt_best.pt` in the same directory. Prefer best for SFT bootstrap;
prefer latest for resuming a paused training run.

Centralized here so future stages (SFT, RL) can bootstrap from a pretrain
checkpoint without each script reinventing the directory walk.
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_GLOB = "ckpt_step_*.pt"
BEST_CHECKPOINT_NAME = "ckpt_best.pt"


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
    """Walk runs newest-first; return the first run's latest checkpoint.

    Falls through runs that have no checkpoints yet (e.g., a crashed run
    that never reached its first periodic save). Returns None only if no
    run under `output_dir` has any checkpoint at all.
    """
    for run in _runs_newest_first(output_dir, run_name):
        ckpt = find_latest_checkpoint(run)
        if ckpt is not None:
            return ckpt
    return None


def find_best_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the run's `ckpt_best.pt` if it exists, else None.

    Use this for SFT bootstrap and any post-training generation that
    should reflect the model's best generalization rather than its
    final-step state.
    """
    p = Path(run_dir) / "checkpoints" / BEST_CHECKPOINT_NAME
    return p if p.exists() else None


def find_best_checkpoint_in(
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path | None:
    """Walk runs newest-first; return the first one with a `ckpt_best.pt`.

    Falls through runs that ran without eval data (no best tracked) or
    that crashed before the first eval. Returns None only if no run has
    a best checkpoint.
    """
    for run in _runs_newest_first(output_dir, run_name):
        ckpt = find_best_checkpoint(run)
        if ckpt is not None:
            return ckpt
    return None


def _runs_newest_first(
    output_dir: str | Path,
    run_name: str | None,
) -> list[Path]:
    out = Path(output_dir)
    if not out.exists():
        return []
    runs = [p for p in out.iterdir() if p.is_dir()]
    if run_name is not None:
        runs = [p for p in runs if p.name.endswith(f"_{run_name}")]
    return sorted(runs, key=lambda p: p.name, reverse=True)
