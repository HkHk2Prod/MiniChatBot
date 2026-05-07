"""Console callback: prints periodic training progress."""

from __future__ import annotations

import time

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS, with hours unbounded (e.g., 100:00:00)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@CALLBACK_REGISTRY.register("console")
class ConsoleCallback(Callback):
    def __init__(self, every: int = 10) -> None:
        self.every = every
        self._t0: float | None = None
        self._last_step: int = 0

    def on_train_start(self, ctx: CallbackContext) -> None:
        self._t0 = time.monotonic()
        self._last_step = ctx.step

        cfg = ctx.config
        n_params = sum(p.numel() for p in ctx.model.parameters())
        device = next(ctx.model.parameters()).device

        lines = [
            "[console] training started",
            f"  run_name : {cfg.run_name}",
            f"  run_dir  : {ctx.run_dir}",
            f"  stage    : {cfg.data.type}",
            f"  device   : {device}",
            f"  seed     : {cfg.seed}",
            f"  model    : {cfg.model.type} "
            f"({n_params / 1e6:.2f}M params, "
            f"L={cfg.model.n_layers} H={cfg.model.n_heads} "
            f"D={cfg.model.d_model} V={cfg.model.vocab_size} "
            f"ctx={cfg.model.max_seq_len})",
            f"  tokenizer: {cfg.tokenizer.type} (path={cfg.tokenizer.path})",
            f"  data     : train={cfg.data.train_path}",
        ]
        if cfg.data.val_path:
            lines.append(f"             val  ={cfg.data.val_path}")
        lines.append(
            f"             seq_len={cfg.data.seq_len}, "
            f"num_workers={cfg.data.num_workers}"
        )
        lines.append(
            f"  trainer  : steps={cfg.trainer.max_steps}, "
            f"batch={cfg.trainer.batch_size}, "
            f"grad_accum={cfg.trainer.grad_accum_steps}, "
            f"precision={cfg.trainer.precision}"
        )
        lines.append(
            f"  optim    : lr={cfg.optim.lr:.2e}, "
            f"warmup={cfg.optim.warmup_steps}, "
            f"sched={cfg.optim.lr_schedule}, "
            f"wd={cfg.optim.weight_decay}"
        )
        print("\n".join(lines))

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.step % self.every != 0 or ctx.loss is None:
            return
        now = time.monotonic()
        elapsed = now - (self._t0 or now)
        steps_done = max(1, ctx.step - self._last_step)
        sps = steps_done / elapsed if elapsed > 0 else 0.0
        parts = [f"step {ctx.step}", f"loss {ctx.loss:.4f}"]
        if ctx.lr is not None:
            parts.append(f"lr {ctx.lr:.2e}")
        if ctx.grad_norm is not None:
            parts.append(f"|g| {ctx.grad_norm:.2f}")
        if ctx.tokens_per_sec is not None:
            parts.append(f"tok/s {ctx.tokens_per_sec:,.0f}")
        parts.append(f"sps {sps:.2f}")
        parts.append(f"elapsed {_fmt_elapsed(elapsed)}")
        print("[console] " + " | ".join(parts))

    def on_eval_end(self, ctx: CallbackContext) -> None:
        if ctx.eval_metrics:
            metrics = " ".join(f"{k}={v:.4f}" for k, v in ctx.eval_metrics.items())
            print(f"[console] eval @ step {ctx.step}: {metrics}")

    def on_train_end(self, ctx: CallbackContext) -> None:
        elapsed = time.monotonic() - (self._t0 or time.monotonic())
        print(
            f"[console] training ended at step {ctx.step} "
            f"(total {_fmt_elapsed(elapsed)})"
        )
