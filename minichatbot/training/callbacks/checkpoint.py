"""Checkpoint callback: periodic saves + best-by-val-loss tracking."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext

BEST_CHECKPOINT_NAME = "ckpt_best.pt"


@CALLBACK_REGISTRY.register("checkpoint")
class CheckpointCallback(Callback):
    """Saves Trainer checkpoints periodically + tracks the best-by-val-loss.

    Three behaviors:
      1. Periodic: every N steps, saves ckpt_step_NNNNNNNN.pt under
         {run_dir}/checkpoints/. Optionally keeps only the last K.
      2. Best: on every on_eval_end where val loss improves over the
         current best, saves to ckpt_best.pt. This file is exempt from
         keep_last_k pruning.
      3. Final periodic save on on_train_end if the most recent step
         wasn't already covered, plus a one-line summary of the best
         model (step + val loss + path).

    Naming: ckpt_step_{step:08d}.pt for periodic, ckpt_best.pt for best.
    Delegates save to ctx.trainer.save_checkpoint so the file format stays
    a Trainer concern, not a callback one.
    """

    def __init__(self, every: int = 1000, keep_last_k: int | None = None) -> None:
        self.every = every
        self.keep_last_k = keep_last_k
        self._best_loss: float | None = None
        self._best_step: int | None = None
        self._best_metrics: dict[str, Any] = {}

    def _ckpt_dir(self, ctx: CallbackContext) -> Path:
        return Path(ctx.run_dir) / "checkpoints"

    def _path_for_step(self, ctx: CallbackContext) -> Path:
        return self._ckpt_dir(ctx) / f"ckpt_step_{ctx.step:08d}.pt"

    def _best_path(self, ctx: CallbackContext) -> Path:
        return self._ckpt_dir(ctx) / BEST_CHECKPOINT_NAME

    def _save(self, ctx: CallbackContext, path: Path) -> Path:
        if ctx.trainer is None:
            raise RuntimeError(
                "CheckpointCallback requires ctx.trainer to be populated. "
                "Run via Trainer.fit() (it sets ctx.trainer automatically)."
            )
        ctx.trainer.save_checkpoint(path)
        return path

    def _prune(self, ctx: CallbackContext) -> None:
        if self.keep_last_k is None or self.keep_last_k <= 0:
            return
        d = self._ckpt_dir(ctx)
        if not d.exists():
            return
        # Glob ONLY step-numbered checkpoints — ckpt_best.pt must survive pruning.
        ckpts = sorted(d.glob("ckpt_step_*.pt"))
        for old in ckpts[: -self.keep_last_k]:
            with contextlib.suppress(OSError):
                old.unlink()

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.step == 0 or ctx.step % self.every != 0:
            return
        path = self._save(ctx, self._path_for_step(ctx))
        ctx.extra["checkpoint_path"] = path
        self._prune(ctx)

    def on_eval_end(self, ctx: CallbackContext) -> None:
        if ctx.trainer is None or not ctx.eval_metrics:
            return
        loss = ctx.eval_metrics.get("loss")
        if loss is None:
            return
        if self._best_loss is None or loss < self._best_loss:
            self._best_loss = float(loss)
            self._best_step = ctx.step
            self._best_metrics = dict(ctx.eval_metrics)
            self._save(ctx, self._best_path(ctx))
            ctx.extra["best_step"] = ctx.step
            ctx.extra["best_loss"] = float(loss)
            ctx.extra["best_path"] = self._best_path(ctx)

    def on_train_end(self, ctx: CallbackContext) -> None:
        # Final periodic save if the latest step wasn't already covered.
        if not self._path_for_step(ctx).exists():
            path = self._save(ctx, self._path_for_step(ctx))
            ctx.extra["checkpoint_path"] = path
            self._prune(ctx)
        # One-line best-model summary. Captured by the logfile tee since
        # logfile.on_train_end fires last (LIFO).
        if self._best_step is not None and self._best_loss is not None:
            ppl = self._best_metrics.get("ppl")
            ppl_str = f", val_ppl={ppl:.2f}" if ppl is not None else ""
            print(
                f"[checkpoint] best model: step {self._best_step}, "
                f"val_loss={self._best_loss:.4f}{ppl_str}, "
                f"path={self._best_path(ctx)}"
            )
        else:
            print(
                "[checkpoint] no best model tracked "
                "(no val loader, or eval never produced metrics)"
            )
