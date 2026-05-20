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
        chat_end_id: int,
        **trainer_kwargs: Any,
    ) -> None:
        super().__init__(**trainer_kwargs)
        if self.tokenizer is None:
            raise ValueError("GRPOTrainer requires a tokenizer (for decoding completions).")
        self.rl_config = rl_config
        self.generator = generator
        self.reward_fn = reward_fn
        self.chat_end_id = chat_end_id

    def _train_step(
        self,
        ctx: CallbackContext,
        train_iter: Iterator[dict[str, Any]],
    ) -> None:
        assert self.tokenizer is not None  # narrowed; __init__ guarantees this
        tokenizer = self.tokenizer  # narrows for the closure below
        t0 = time.monotonic()
        gen_tokens = 0.0

        def _micro(it: Iterator[Any]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
            nonlocal gen_tokens
            prompts = next(it)
            rollout = collect_rollouts(
                model=self.model,
                generator=self.generator,
                reward_fn=self.reward_fn,
                tokenizer=tokenizer,
                prompts=prompts["prompt_ids"],
                references=prompts["reference"],
                group_size=self.rl_config.group_size,
                max_new_tokens=self.rl_config.max_new_tokens,
                chat_end_id=self.chat_end_id,
                pad_id=tokenizer.pad_id,
                device=self.device,
                normalize_advantage_std=self.rl_config.normalize_advantage_std,
            )
            gen_tokens += float(rollout.batch["loss_mask"].sum().item())
            return rollout.batch, {
                "reward_mean": rollout.reward_mean,
                "solve_rate": rollout.solve_rate,
                "gen_len_mean": rollout.gen_len_mean,
            }

        self._run_accum_step(ctx, train_iter, _micro)

        step_dt = time.monotonic() - t0
        ctx.tokens_per_sec = gen_tokens / step_dt if step_dt > 0 else None
