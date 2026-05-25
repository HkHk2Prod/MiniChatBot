"""Loss-curve callback: writes a train/val loss PNG at end-of-training.

Accumulates the per-step train loss (`on_step_end`) and the periodic
validation loss (`on_eval_end`) in memory, then at `on_train_end` renders
a single `loss_curve.png` next to the checkpoints. The point is the
overfit/underfit read at a glance: train and val falling together →
keep going (undertrained); val flattening or turning up while train keeps
dropping → overfitting; both flat and high → stuck. Mirrors the
`benchmark` callback's "one self-documenting artifact per run dir" idea,
but for the loss trajectory instead of sample generations.

In-memory rather than re-parsing metrics.jsonl so the callback is
self-contained (no dependency on the jsonl callback being configured)
and so a fresh run_dir's plot only ever shows that run's own curve.

matplotlib is an optional dependency (the `[plot]` extra). It's
lazy-imported here and, like the benchmark callback, a failure NEVER
crashes a run that otherwise succeeded — a missing matplotlib just logs
an install hint and skips. RL has no validation pass, so the val series
is empty there and the plot shows train loss alone.

Configure in your run's `callbacks:` block (no params required):

    - type: lossplot
      params:
        filename: loss_curve.png   # output name, written under run_dir
        smooth: 0.9                # EMA weight for the train curve (0 = off)
"""

from __future__ import annotations

from pathlib import Path

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext


@CALLBACK_REGISTRY.register("lossplot")
class LossPlotCallback(Callback):
    """Collect train/val loss during training; plot it at the end."""

    def __init__(self, filename: str = "loss_curve.png", smooth: float = 0.9) -> None:
        if not 0.0 <= smooth < 1.0:
            raise ValueError(f"smooth must be in [0, 1), got {smooth}")
        self.filename = filename
        self.smooth = smooth
        # Parallel (step, loss) histories — train logged every step, val
        # only when the eval callback publishes metrics.
        self._train_steps: list[int] = []
        self._train_loss: list[float] = []
        self._val_steps: list[int] = []
        self._val_loss: list[float] = []

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.loss is None:
            return
        self._train_steps.append(ctx.step)
        self._train_loss.append(float(ctx.loss))

    def on_eval_end(self, ctx: CallbackContext) -> None:
        # The eval callback reports val loss under "loss"; ignore evals that
        # don't carry one (keeps this robust to other eval-metric producers).
        if not ctx.eval_metrics or "loss" not in ctx.eval_metrics:
            return
        self._val_steps.append(ctx.step)
        self._val_loss.append(float(ctx.eval_metrics["loss"]))

    def on_train_end(self, ctx: CallbackContext) -> None:
        # Observer contract: a plotting failure must not sink a finished run.
        try:
            self._render(ctx)
        except Exception as exc:  # noqa: BLE001
            print(f"[lossplot] skipped: {exc}")

    def _ema(self, values: list[float]) -> list[float]:
        """Exponential moving average; per-step train loss is noisy enough
        that the raw line buries the trend on a long run."""
        out: list[float] = []
        avg: float | None = None
        for v in values:
            avg = v if avg is None else self.smooth * avg + (1.0 - self.smooth) * v
            out.append(avg)
        return out

    def _render(self, ctx: CallbackContext) -> None:
        if not self._train_steps and not self._val_steps:
            print("[lossplot] skipped: no loss recorded.")
            return

        try:
            import matplotlib
        except ImportError:
            print(
                "[lossplot] skipped: matplotlib not installed. "
                'Install with: pip install -e ".[plot]"'
            )
            return
        # Headless-safe: pick the non-interactive backend before pyplot binds
        # one, so this works on a server with no display during a long run.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stage = ctx.config.data.type
        fig, ax = plt.subplots(figsize=(9, 5.5))

        if self._train_steps:
            if self.smooth > 0.0 and len(self._train_loss) > 1:
                # Raw faint, smoothed solid — keep both so spikes are visible
                # but the trend reads cleanly.
                ax.plot(
                    self._train_steps,
                    self._train_loss,
                    color="tab:blue",
                    alpha=0.25,
                    linewidth=0.8,
                )
                ax.plot(
                    self._train_steps,
                    self._ema(self._train_loss),
                    color="tab:blue",
                    linewidth=1.8,
                    label=f"train (EMA {self.smooth:g})",
                )
            else:
                ax.plot(
                    self._train_steps,
                    self._train_loss,
                    color="tab:blue",
                    linewidth=1.5,
                    label="train",
                )

        if self._val_steps:
            ax.plot(
                self._val_steps,
                self._val_loss,
                color="tab:orange",
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="val",
            )
            # The min-val point is the checkpoint that actually ships
            # (ckpt_best.pt) — mark it so the plot answers "where was best?".
            best_i = min(range(len(self._val_loss)), key=self._val_loss.__getitem__)
            bx, by = self._val_steps[best_i], self._val_loss[best_i]
            ax.scatter([bx], [by], color="tab:red", zorder=5, s=40)
            ax.annotate(
                f"best val {by:.3f} @ {bx}",
                xy=(bx, by),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="tab:red",
            )

        ax.set_xlabel("step")
        ax.set_ylabel("loss (cross-entropy)")
        ax.set_title(f"{ctx.config.run_name} — {stage} loss")
        ax.grid(True, alpha=0.3)
        if self._train_steps or self._val_steps:
            ax.legend(loc="upper right")

        out_path = Path(ctx.run_dir) / self.filename
        print(f"[lossplot] writing {out_path}")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
