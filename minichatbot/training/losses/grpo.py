"""GRPO policy-gradient surrogate loss.

Consumes a batch produced by `minichatbot.rl.collect_rollouts`:
    input_ids  (B, T)   prompt + sampled completion, right-padded
    loss_mask  (B, T)   1.0 on completion tokens, 0.0 on prompt/pad
    advantages (B,)     group-normalized reward advantage per sequence

Loss = mean over completion tokens of  -advantage * log p(token).

Because the RL trainer takes a single gradient step per batch of *fresh*
on-policy rollouts, the importance-sampling ratio between the sampling
policy and the policy being updated is exactly 1, so PPO's clipped
objective collapses to this plain advantage-weighted log-likelihood —
no `old_logprobs`, no clip range, no second forward pass. (If multi-epoch
updates per rollout are ever added, this is where the ratio + clip go.)

A per-token mean (not per-sequence) keeps long correct completions from
dominating short ones; it matches the "token-level loss" choice in the
GRPO/Dr.GRPO line of work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minichatbot.model.base import ModelOutput
from minichatbot.training.losses import LOSS_REGISTRY
from minichatbot.training.losses.base import Loss


@LOSS_REGISTRY.register("grpo")
class GRPOLoss(Loss):
    def forward(
        self,
        output: ModelOutput,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Predicting token i uses logits at position i-1, so align
        # logits[:, :-1] with input_ids[:, 1:] (and the mask likewise).
        logits = output.logits[:, :-1, :]            # (B, T-1, V)
        targets = batch["input_ids"][:, 1:]          # (B, T-1)
        mask = batch["loss_mask"][:, 1:]             # (B, T-1)
        advantages = batch["advantages"].unsqueeze(1)  # (B, 1)

        b, t_minus_1, vocab = logits.shape
        # cross_entropy == -log p(target). Flatten so this is one fused
        # kernel; reduction="none" to apply the mask + advantage ourselves.
        # (Under autocast, cross_entropy already upcasts internally.)
        neg_logp = F.cross_entropy(
            logits.reshape(b * t_minus_1, vocab),
            targets.reshape(b * t_minus_1),
            reduction="none",
        ).reshape(b, t_minus_1)

        # -advantage * log p  ==  advantage * (-log p)  ==  advantage * neg_logp
        per_token = advantages * neg_logp * mask
        return per_token.sum() / mask.sum().clamp(min=1.0)
