"""Checkpoint callback: periodic Trainer.save_checkpoint() calls."""

from __future__ import annotations

from pathlib import Path

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext


@CALLBACK_REGISTRY.register("checkpoint")
class CheckpointCallback(Callback):
    """Saves a Trainer checkpoint every N steps; optionally retains only
    the last K. Always saves a final checkpoint on `on_train_end` (skipped
    if the last periodic save already produced one for the current step).

    Naming: ckpt_step_{step:08d}.pt under {run_dir}/checkpoints/.
    Delegates the actual save to `ctx.trainer.save_checkpoint(path)` so
    the file format stays a Trainer concern, not a callback one.
    """

    def __init__(self, every: int = 1000, keep_top_k: int | None = None) -> None:
        self.every = every
        self.keep_top_k = keep_top_k

    def _ckpt_dir(self, ctx: CallbackContext) -> Path:
        return Path(ctx.run_dir) / "checkpoints"

    def _path_for_step(self, ctx: CallbackContext) -> Path:
        return self._ckpt_dir(ctx) / f"ckpt_step_{ctx.step:08d}.pt"

    def _save(self, ctx: CallbackContext) -> Path:
        if ctx.trainer is None:
            raise RuntimeError(
                "CheckpointCallback requires ctx.trainer to be populated. "
                "Run via Trainer.fit() (it sets ctx.trainer automatically)."
            )
        path = self._path_for_step(ctx)
        ctx.trainer.save_checkpoint(path)
        return path

    def _prune(self, ctx: CallbackContext) -> None:
        if self.keep_top_k is None or self.keep_top_k <= 0:
            return
        d = self._ckpt_dir(ctx)
        if not d.exists():
            return
        ckpts = sorted(d.glob("ckpt_step_*.pt"))
        for old in ckpts[: -self.keep_top_k]:
            try:
                old.unlink()
            except OSError:
                pass

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.step == 0 or ctx.step % self.every != 0:
            return
        path = self._save(ctx)
        ctx.extra["checkpoint_path"] = path
        self._prune(ctx)

    def on_train_end(self, ctx: CallbackContext) -> None:
        if self._path_for_step(ctx).exists():
            return
        path = self._save(ctx)
        ctx.extra["checkpoint_path"] = path
        self._prune(ctx)
