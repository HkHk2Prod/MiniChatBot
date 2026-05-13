"""GRPO trainer: same lifecycle as `Trainer`, different step.

`Trainer._train_step` is supervised — pull a labelled batch, forward,
loss, backward. The RL step instead, per training step:

    1. Pull a batch of *prompts* from the loader.
    2. Sample `group_size` completions per prompt, score each with the
       reward fn, mean-center within the group -> advantages, and pack
       (prompt + completion) sequences into a training batch
       (`minichatbot.rl.collect_rollouts`).
    3. One forward + `GRPOLoss` + backward on that batch (on-policy, so a
       single step — see GRPOLoss for why no PPO ratio/clip is needed).

Everything else — grad accumulation, grad clipping, AMP/GradScaler,
optimizer/scheduler stepping, callbacks, checkpointing — is inherited
unchanged. Each grad-accum micro-step is its own rollout collection, so
`grad_accum_steps` simply means "average the policy gradient over this
many prompt batches before stepping".

Reward / generation stats land in `ctx.extra` (`reward_mean`,
`solve_rate`, `gen_len_mean`) so console/jsonl callbacks can surface
them; `ctx.loss` is the surrogate loss, which is near-zero and not a
progress signal by itself — watch `solve_rate`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import torch

from minichatbot.config import RLConfig
from minichatbot.inference.generator import Generator
from minichatbot.rl.rewards.base import Reward
from minichatbot.rl.rollout import collect_rollouts
from minichatbot.training.callbacks.base import CallbackContext
from minichatbot.training.trainer import Trainer


class GRPOTrainer(Trainer):
    def __init__(
        self,
        *,
        rl_config: RLConfig,
        generator: Generator,
        reward_fn: Reward,
        eos_id: int,
        **trainer_kwargs: Any,
    ) -> None:
        super().__init__(**trainer_kwargs)
        if self.tokenizer is None:
            raise ValueError("GRPOTrainer requires a tokenizer (for decoding completions).")
        self.rl_config = rl_config
        self.generator = generator
        self.reward_fn = reward_fn
        self.eos_id = eos_id

    def _train_step(
        self,
        ctx: CallbackContext,
        train_iter: Iterator[dict[str, Any]],
    ) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum = self.config.grad_accum_steps
        t0 = time.monotonic()
        total_loss = 0.0
        reward_mean_sum = 0.0
        solve_rate_sum = 0.0
        gen_len_sum = 0.0
        gen_tokens = 0.0
        last_batch: dict[str, torch.Tensor] = {}

        for _ in range(accum):
            prompts = next(train_iter)
            rollout = collect_rollouts(
                model=self.model,
                generator=self.generator,
                reward_fn=self.reward_fn,
                tokenizer=self.tokenizer,
                prompts=prompts["prompt_ids"],
                references=prompts["reference"],
                group_size=self.rl_config.group_size,
                max_new_tokens=self.rl_config.max_new_tokens,
                eos_id=self.eos_id,
                pad_id=self.tokenizer.pad_id,
                device=self.device,
                normalize_advantage_std=self.rl_config.normalize_advantage_std,
            )
            batch = rollout.batch
            last_batch = batch
            with self.autocast():
                output = self.model(batch["input_ids"])
                actual_loss = self.loss(output, batch)
                scaled = actual_loss / accum
            self._backward(scaled)

            total_loss += float(actual_loss.item())
            reward_mean_sum += rollout.reward_mean
            solve_rate_sum += rollout.solve_rate
            gen_len_sum += rollout.gen_len_mean
            gen_tokens += float(batch["loss_mask"].sum().item())

        self._fire("on_backward_end", ctx)
        grad_norm = self._optimizer_step()

        step_dt = time.monotonic() - t0
        ctx.batch = last_batch
        ctx.loss = total_loss / accum
        ctx.grad_norm = grad_norm
        ctx.lr = float(self.scheduler.get_last_lr()[0])
        ctx.tokens_per_sec = gen_tokens / step_dt if step_dt > 0 else None
        ctx.extra["reward_mean"] = reward_mean_sum / accum
        ctx.extra["solve_rate"] = solve_rate_sum / accum
        ctx.extra["gen_len_mean"] = gen_len_sum / accum
