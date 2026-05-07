"""Weights & Biases callback (optional dependency)."""

from __future__ import annotations

import dataclasses
from typing import Any

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext


@CALLBACK_REGISTRY.register("wandb")
class WandbCallback(Callback):
    def __init__(
        self,
        project: str = "minichatbot",
        entity: str | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self._wandb: Any = None
        self._run: Any = None

    def on_train_start(self, ctx: CallbackContext) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is required for WandbCallback. "
                'Install with: pip install -e ".[wandb]"'
            ) from e
        self._wandb = wandb
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=ctx.config.run_name,
            dir=str(ctx.run_dir),
            config=dataclasses.asdict(ctx.config),
        )

    def on_step_end(self, ctx: CallbackContext) -> None:
        if self._wandb is None or ctx.loss is None:
            return
        log: dict[str, Any] = {"train/loss": ctx.loss}
        for k in ("lr", "grad_norm", "tokens_per_sec"):
            v = getattr(ctx, k)
            if v is not None:
                log[f"train/{k}"] = v
        self._wandb.log(log, step=ctx.step)

    def on_eval_end(self, ctx: CallbackContext) -> None:
        if self._wandb is None or not ctx.eval_metrics:
            return
        self._wandb.log(
            {f"eval/{k}": v for k, v in ctx.eval_metrics.items()},
            step=ctx.step,
        )

    def on_train_end(self, ctx: CallbackContext) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
