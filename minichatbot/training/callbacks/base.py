"""Base class and context for training callbacks.

Callbacks are observers and side-effect agents only — they MUST NOT
mutate training state (loss, gradients, model weights, optimizer or
scheduler state). Behavioral changes belong in a Trainer subclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from minichatbot.config import Config


@dataclass
class CallbackContext:
    """Per-event state passed to every callback.

    Treat this as the public contract of the callback surface: adding
    fields is additive and safe; renaming or removing them breaks every
    callback subclass. Per-step / per-eval fields default to None so the
    same dataclass serves every event without per-event variants.

    Trainer-state fields (model, optimizer, scheduler, val_loader,
    loss_fn, tokenizer) are populated once at construction and remain
    static across events. Per-step and per-eval fields are repopulated
    by the trainer before each event.
    """

    step: int
    epoch: int
    run_dir: Path
    config: Config
    model: nn.Module

    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any | None = None
    val_loader: Any | None = None
    loss_fn: Any | None = None
    tokenizer: Any | None = None
    trainer: Any | None = None

    batch: dict[str, torch.Tensor] | None = None
    loss: float | None = None
    grad_norm: float | None = None
    lr: float | None = None
    tokens_per_sec: float | None = None

    eval_metrics: dict[str, float] | None = None

    extra: dict[str, Any] = field(default_factory=dict)


class Callback:
    """Base class for all training callbacks.

    All event hooks default to no-op so subclasses only override the
    events they care about. There is no `@abstractmethod` here on
    purpose: a callback that subscribes to one event is fine.
    """

    def on_train_start(self, ctx: CallbackContext) -> None: ...
    def on_epoch_start(self, ctx: CallbackContext) -> None: ...
    def on_step_start(self, ctx: CallbackContext) -> None: ...
    def on_backward_end(self, ctx: CallbackContext) -> None: ...
    def on_step_end(self, ctx: CallbackContext) -> None: ...
    def on_eval_start(self, ctx: CallbackContext) -> None: ...
    def on_eval_end(self, ctx: CallbackContext) -> None: ...
    def on_checkpoint(self, ctx: CallbackContext) -> None: ...
    def on_epoch_end(self, ctx: CallbackContext) -> None: ...
    def on_train_end(self, ctx: CallbackContext) -> None: ...
