"""DPO trainer: supervised lifecycle, reference-anchored preference step.

Like `GRPOTrainer`, it keeps `Trainer`'s lifecycle and reuses
`_run_accum_step`, but its micro-step does NOT sample. Instead, per batch
of multiple-choice candidate rows (from the `mc` collator) it:

    1. Runs the frozen reference model (no grad) over the candidate rows
       and stashes the per-row log-probs in `batch["ref_logp"]`.
    2. Lets `_run_accum_step` do the policy forward + `DPOLoss`, which
       turns (policy − reference) log-ratios into an N-way cross-entropy
       toward the gold candidate.

The reference is a frozen snapshot of the initial (from-pretrained)
policy, built in `runner._build_dpo_trainer`. `DPOLoss.last_acc` /
`last_margin` are surfaced on `ctx.extra` so console/jsonl callbacks show
the multiple-choice accuracy and gold-vs-distractor margin climbing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import torch

from minichatbot.model.base import LanguageModel
from minichatbot.training.callbacks.base import CallbackContext
from minichatbot.training.losses.dpo import sequence_logp
from minichatbot.training.trainer import Trainer


class DPOTrainer(Trainer):
    def __init__(self, *, ref_model: LanguageModel, **trainer_kwargs: Any) -> None:
        super().__init__(**trainer_kwargs)
        self.ref_model = ref_model

    def _train_step(
        self,
        ctx: CallbackContext,
        train_iter: Iterator[dict[str, Any]],
    ) -> None:
        t0 = time.monotonic()
        scored_tokens = 0.0

        def _micro(it: Iterator[Any]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
            nonlocal scored_tokens
            batch = next(it)
            batch = {
                k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            # Reference log-probs: constant w.r.t. the policy update, so
            # compute once here (no grad) and hand them to DPOLoss via batch.
            with torch.no_grad(), self.autocast():
                ref_logits = self.ref_model(batch["input_ids"]).logits
            batch["ref_logp"] = sequence_logp(
                ref_logits, batch["input_ids"], batch["lengths"], batch["n_conts"]
            )
            scored_tokens += float(batch["n_conts"].sum().item())
            return batch, {}

        self._run_accum_step(ctx, train_iter, _micro)

        # Surface the monitoring scalars DPOLoss recorded on its last forward.
        ctx.extra["mc_acc"] = float(getattr(self.loss, "last_acc", 0.0))
        ctx.extra["margin"] = float(getattr(self.loss, "last_margin", 0.0))

        step_dt = time.monotonic() - t0
        ctx.tokens_per_sec = scored_tokens / step_dt if step_dt > 0 else None
