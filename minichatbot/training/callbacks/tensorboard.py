"""TensorBoard callback (optional dependency).

Lazy-imports `torch.utils.tensorboard` in on_train_start so the rest of
the project doesn't require tensorboard installed unless the user
actually configures this callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext


@CALLBACK_REGISTRY.register("tensorboard")
class TensorBoardCallback(Callback):
    def __init__(self, subdir: str = "tb") -> None:
        self.subdir = subdir
        self._writer: Any | None = None

    def on_train_start(self, ctx: CallbackContext) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "tensorboard is required for TensorBoardCallback. "
                'Install with: pip install -e ".[tensorboard]"'
            ) from e
        path = Path(ctx.run_dir) / self.subdir
        path.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(path))

    def on_step_end(self, ctx: CallbackContext) -> None:
        if self._writer is None or ctx.loss is None:
            return
        self._writer.add_scalar("train/loss", ctx.loss, ctx.step)
        if ctx.lr is not None:
            self._writer.add_scalar("train/lr", ctx.lr, ctx.step)
        if ctx.grad_norm is not None:
            self._writer.add_scalar("train/grad_norm", ctx.grad_norm, ctx.step)
        if ctx.tokens_per_sec is not None:
            self._writer.add_scalar("train/tokens_per_sec", ctx.tokens_per_sec, ctx.step)

    def on_eval_end(self, ctx: CallbackContext) -> None:
        if self._writer is None or not ctx.eval_metrics:
            return
        for k, v in ctx.eval_metrics.items():
            self._writer.add_scalar(f"eval/{k}", v, ctx.step)

    def on_train_end(self, ctx: CallbackContext) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
