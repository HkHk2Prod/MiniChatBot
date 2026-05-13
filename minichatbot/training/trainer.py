"""Trainer for autoregressive language-model training.

One trainer covers pretrain and SFT (just different Loss + Collator
+ Dataset). The RL stage subclasses it — see
`minichatbot.training.rl_trainer.GRPOTrainer`, which keeps this
lifecycle but swaps the supervised forward/backward for
sample -> reward -> policy-gradient.

Lifecycle:
    on_train_start
      [ on_step_start
        N x ( forward -> loss / N -> backward )    # gradient accumulation
        on_backward_end
        grad_clip + optimizer.step + scheduler.step + zero_grad
        on_step_end
        if any callback set ctx.eval_metrics: on_eval_end (then clear) ]
    on_train_end                            # fired in REVERSE order (LIFO)

Checkpointing:
    save_checkpoint(path) bundles model_config + model + optimizer +
    scheduler + step + full_config (+ optional extra). load_checkpoint
    restores all of those into an already-constructed Trainer. The
    saved model_config + model keys are also LanguageModel.load-compatible.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from minichatbot.config import Config, TrainerConfig
from minichatbot.model.base import LanguageModel
from minichatbot.training.callbacks.base import Callback, CallbackContext
from minichatbot.training.losses.base import Loss
from minichatbot.utils.io import atomic_torch_save


def _resolve_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown precision: {precision!r}")


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        full_config: Config,
        model: LanguageModel,
        loss: Loss,
        optimizer: Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        callbacks: list[Callback],
        run_dir: Path,
        device: torch.device,
        val_loader: DataLoader | None = None,
        tokenizer: Any | None = None,
        startup_warnings: list[str] | None = None,
    ) -> None:
        self.config = config
        self.full_config = full_config
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks
        self.run_dir = run_dir
        self.device = device
        self.tokenizer = tokenizer

        # Vocab consistency: a model with vocab_size != tokenizer.vocab_size
        # silently produces wrong outputs (out-of-range token ids → garbage,
        # or unused embedding rows → wasted params). Fail loudly here rather
        # than discover it via a NaN loss or a generation that decodes weird.
        if tokenizer is not None and model.cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"vocab_size mismatch: model.cfg.vocab_size={model.cfg.vocab_size}, "
                f"tokenizer.vocab_size={tokenizer.vocab_size}. They must match. "
                "Update model.vocab_size in your YAML config to match the tokenizer, "
                "or retrain the tokenizer with a different --vocab-size."
            )

        self.dtype = _resolve_dtype(config.precision)
        self.use_autocast = config.precision != "fp32"
        self.scaler: GradScaler | None = (
            GradScaler("cuda")
            if config.precision == "fp16" and device.type == "cuda"
            else None
        )

        # Tracks the most recently completed training step. Persisted by
        # save_checkpoint and resumed by load_checkpoint; fit() iterates
        # range(self.step + 1, max_steps + 1), so resumption is automatic.
        self.step: int = 0

        # Messages built outside the trainer (e.g. YAML/checkpoint model
        # config reconciliation) that should appear in log.txt — replayed
        # in fit() AFTER on_train_start so the LogFile callback's stdout
        # tee is already installed.
        self._startup_warnings: list[str] = list(startup_warnings or [])

    def fit(self) -> None:
        ctx = self._make_ctx()
        self._fire("on_train_start", ctx)
        # Replay deferred startup warnings here, not at the call site that
        # built them — by now the LogFile callback's stdout tee is in
        # place, so banners end up in log.txt as well as the console.
        for warning in self._startup_warnings:
            print(warning, flush=True)
        # Allow callbacks to publish baseline eval metrics from on_train_start
        # (e.g., eval-at-step-0 to show the pre-training reference loss).
        if ctx.eval_metrics is not None:
            self._fire("on_eval_end", ctx)
            ctx.eval_metrics = None
        try:
            train_iter = self._cycling(self.train_loader)
            for step in range(self.step + 1, self.config.max_steps + 1):
                self.step = step
                ctx.step = step
                self._fire("on_step_start", ctx)
                self._train_step(ctx, train_iter)
                self._fire("on_step_end", ctx)
                if ctx.eval_metrics is not None:
                    self._fire("on_eval_end", ctx)
                    ctx.eval_metrics = None
        finally:
            self._fire("on_train_end", ctx)

    def save_checkpoint(
        self,
        path: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save full training state: model + config + optimizer + scheduler + step.

        Format is a superset of LanguageModel.save() — the saved file can
        be loaded as a model via LanguageModel.load() or load_model().
        """
        state: dict[str, Any] = {
            "model_config": dataclasses.asdict(self.model.cfg),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "full_config": dataclasses.asdict(self.full_config),
        }
        if extra:
            state["extra"] = extra
        atomic_torch_save(state, path)

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
        *,
        preloaded_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Restore model / optimizer / scheduler / step into this Trainer.

        Returns the `extra` dict from the checkpoint (or empty dict).
        Caller is responsible for ensuring the trainer was constructed
        with a model architecture compatible with the checkpoint.

        Pass ``preloaded_state`` to reuse a state dict the caller already
        loaded (e.g., for upfront model_config reconciliation), avoiding
        a redundant torch.load of a large checkpoint.
        """
        if map_location is None:
            map_location = self.device
        if preloaded_state is not None:
            state = preloaded_state
        else:
            state = torch.load(path, map_location=map_location, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state.get("step", 0))
        return dict(state.get("extra", {}))

    def _train_step(
        self,
        ctx: CallbackContext,
        train_iter: Iterator[dict[str, torch.Tensor]],
    ) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum = self.config.grad_accum_steps
        t0 = time.monotonic()
        total_loss = 0.0
        last_batch: dict[str, torch.Tensor] = {}

        for _ in range(accum):
            batch = next(train_iter)
            batch = {
                k: v.to(self.device, non_blocking=True) for k, v in batch.items()
            }
            last_batch = batch
            with self.autocast():
                output = self.model(batch["input_ids"])
                actual_loss = self.loss(output, batch)
                scaled = actual_loss / accum
            self._backward(scaled)
            total_loss += float(actual_loss.item())

        self._fire("on_backward_end", ctx)
        grad_norm = self._optimizer_step()

        step_dt = time.monotonic() - t0
        seq_len = last_batch["input_ids"].size(1)
        tokens = self.config.batch_size * accum * seq_len

        ctx.batch = last_batch
        ctx.loss = total_loss / accum
        ctx.grad_norm = grad_norm
        ctx.lr = float(self.scheduler.get_last_lr()[0])
        ctx.tokens_per_sec = tokens / step_dt if step_dt > 0 else None

    def _backward(self, scaled_loss: torch.Tensor) -> None:
        """Backward a (already grad-accum-scaled) loss, through the GradScaler
        when fp16 is active. Shared by Trainer._train_step and subclasses
        (e.g., GRPOTrainer) so the AMP plumbing lives in exactly one place."""
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _optimizer_step(self) -> float | None:
        """Unscale (fp16) -> grad-clip -> optimizer.step -> scheduler.step.

        Returns the pre-clip grad norm if grad_clip is set, else None.
        Call once per training step, after all grad-accum backwards.
        """
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm: float | None = None
        if self.config.grad_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            )
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        return grad_norm

    def autocast(self):
        """Return the autocast context that wraps every train forward.

        Public so callbacks (eval, sample, etc.) can run their own
        forwards under the same precision regime as training — keeps
        eval loss comparable to train loss and avoids fp32-on-bf16-model
        slowdowns.
        """
        if not self.use_autocast:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def _make_ctx(self) -> CallbackContext:
        return CallbackContext(
            step=self.step,
            epoch=0,
            run_dir=self.run_dir,
            config=self.full_config,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            val_loader=self.val_loader,
            loss_fn=self.loss,
            tokenizer=self.tokenizer,
            trainer=self,
        )

    def _fire(self, event: str, ctx: CallbackContext) -> None:
        # Teardown events fire in reverse order (LIFO) so resources opened
        # by earlier callbacks (e.g., LogFileCallback's stdout tee) outlive
        # later callbacks' final prints.
        callbacks = (
            reversed(self.callbacks) if event == "on_train_end" else self.callbacks
        )
        for cb in callbacks:
            getattr(cb, event)(ctx)

    @staticmethod
    def _cycling(loader: DataLoader) -> Iterator[Any]:
        while True:
            yield from loader
