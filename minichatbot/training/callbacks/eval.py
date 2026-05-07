"""Eval callback: runs validation loss every N steps."""

from __future__ import annotations

import math
import time

import torch

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext
from minichatbot.utils.torch_helpers import eval_mode


@CALLBACK_REGISTRY.register("eval")
class EvalCallback(Callback):
    """Runs the loss on a held-out loader every N steps and stuffs the
    result into ctx.eval_metrics so downstream callbacks (Console, Jsonl,
    TensorBoard, W&B) can report it.

    Reads the val DataLoader and Loss from `ctx.val_loader` /
    `ctx.loss_fn`, both populated by the Trainer at construction time.
    Silently skips when either is None.
    """

    def __init__(
        self,
        every: int = 500,
        max_batches: int | None = None,
        eval_at_start: bool = True,
    ) -> None:
        self.every = every
        self.max_batches = max_batches
        self.eval_at_start = eval_at_start

    def on_train_start(self, ctx: CallbackContext) -> None:
        # Baseline eval before any training so the loss trajectory has a
        # reference point. The Trainer dispatches on_eval_end after
        # on_train_start when eval_metrics is set, so downstream callbacks
        # (console/jsonl/tensorboard) will report it normally.
        if not self.eval_at_start:
            return
        if ctx.val_loader is None or ctx.loss_fn is None:
            return
        self._run(ctx)

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.val_loader is None or ctx.loss_fn is None:
            return
        if ctx.step == 0 or ctx.step % self.every != 0:
            return
        self._run(ctx)
        # No final eval in on_train_end: doing so would race the cleanup of
        # other callbacks (jsonl/tensorboard/wandb close their writers in
        # on_train_end). Align max_steps with `every` to capture a final eval.

    def _run(self, ctx: CallbackContext) -> None:
        model = ctx.model
        with eval_mode(model):
            total = 0.0
            n = 0
            device = next(model.parameters()).device
            t0 = time.monotonic()
            with torch.no_grad():
                for i, batch in enumerate(ctx.val_loader):
                    if self.max_batches is not None and i >= self.max_batches:
                        break
                    batch = {
                        k: v.to(device, non_blocking=True) for k, v in batch.items()
                    }
                    output = model(batch["input_ids"])
                    loss = ctx.loss_fn(output, batch)
                    total += float(loss.item())
                    n += 1
            elapsed = time.monotonic() - t0
            avg = total / max(1, n)
            ctx.eval_metrics = {
                "loss": avg,
                "ppl": float(math.exp(min(avg, 20.0))),
                "n_batches": n,
                "elapsed_s": elapsed,
            }
