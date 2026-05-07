"""JSON-Lines callback: appends one JSON line per logged event.

Source-of-truth metrics log. Pandas/matplotlib can replay it for plots
without needing TensorBoard or W&B.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext


@CALLBACK_REGISTRY.register("jsonl")
class JsonlCallback(Callback):
    def __init__(self, filename: str = "metrics.jsonl") -> None:
        self.filename = filename
        self._fh: IO[str] | None = None

    def on_train_start(self, ctx: CallbackContext) -> None:
        path = Path(ctx.run_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1, encoding="utf-8")

    def _write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record) + "\n")

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.loss is None:
            return
        record: dict[str, Any] = {
            "event": "step",
            "step": ctx.step,
            "loss": ctx.loss,
        }
        for k in ("lr", "grad_norm", "tokens_per_sec"):
            v = getattr(ctx, k)
            if v is not None:
                record[k] = v
        self._write(record)

    def on_eval_end(self, ctx: CallbackContext) -> None:
        if not ctx.eval_metrics:
            return
        self._write({"event": "eval", "step": ctx.step, **ctx.eval_metrics})

    def on_checkpoint(self, ctx: CallbackContext) -> None:
        self._write({"event": "checkpoint", "step": ctx.step})

    def on_train_end(self, ctx: CallbackContext) -> None:
        self._write({"event": "train_end", "step": ctx.step})
        if self._fh is not None:
            self._fh.close()
            self._fh = None
