"""Sample completions, score them, and pack the result into a training batch.

`collect_rollouts` is the bridge between the inference layer (sampling
completions with `Generator`) and the RL loss (`GRPOLoss`):

    for each prompt:
        sample `group_size` completions (shared length budget, same prompt
            replicated -> a clean (G, T) batch with no left-padding needed)
        score each completion with the reward fn
        advantage_i = reward_i - mean(group)         # GRPO baseline
                    [ / (std(group) + eps) ]         # optional std-normalize
    pad all (prompt + completion) sequences into one (B, T) batch with a
    `loss_mask` that is 1.0 on completion tokens (the ones the policy
    actually produced) and 0.0 on prompt and pad tokens.

The returned `batch` dict is exactly what `GRPOLoss.forward` expects.
Generation stops at `eos_id` (the chat `<|im_end|>` token): finished
sequences get eos-padded by `Generator`, and we trim each completion at
its first `eos_id` (kept, so the policy still learns to emit it).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minichatbot.inference.generator import Generator
from minichatbot.model.base import LanguageModel
from minichatbot.rl.rewards.base import Reward
from minichatbot.tokenizer.base import Tokenizer


@dataclass
class RolloutResult:
    """Output of one rollout collection.

    `batch` holds CUDA/CPU tensors ready for `GRPOLoss`:
        input_ids  (B, T)  prompt + completion, right-padded with pad_id
        loss_mask  (B, T)  1.0 on completion tokens, 0.0 on prompt/pad
        advantages (B,)    group-normalized reward advantage per sequence
    The scalar fields are for logging only.
    """

    batch: dict[str, torch.Tensor]
    reward_mean: float
    reward_std: float
    solve_rate: float          # fraction of sampled completions with reward > 0
    gen_len_mean: float        # mean completion length in tokens
    n_samples: int             # B = n_prompts * group_size


@torch.no_grad()
def collect_rollouts(
    *,
    model: LanguageModel,
    generator: Generator,
    reward_fn: Reward,
    tokenizer: Tokenizer,
    prompts: list[torch.Tensor],
    references: list[str],
    group_size: int,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    device: torch.device,
    normalize_advantage_std: bool,
) -> RolloutResult:
    seqs: list[list[int]] = []
    prompt_lens: list[int] = []
    advantages: list[float] = []
    rewards: list[float] = []

    for prompt_ids, reference in zip(prompts, references):
        tp = int(prompt_ids.size(0))
        replicated = prompt_ids.to(device).unsqueeze(0).expand(group_size, tp)
        out = generator.generate(model, replicated, max_new_tokens=max_new_tokens)
        rows = out.tolist()

        group_rewards: list[float] = []
        group_seqs: list[list[int]] = []
        for row in rows:
            completion = row[tp:]
            if eos_id in completion:
                completion = completion[: completion.index(eos_id) + 1]
            if not completion:                      # model produced nothing
                completion = [eos_id]
            text = tokenizer.decode(completion, include_special=False)
            group_rewards.append(float(reward_fn(text, reference)))
            group_seqs.append(row[:tp] + completion)

        group = torch.tensor(group_rewards, dtype=torch.float32)
        adv = group - group.mean()
        if normalize_advantage_std:
            adv = adv / (group.std(unbiased=False) + 1e-6)
        for seq, a, r in zip(group_seqs, adv.tolist(), group_rewards):
            seqs.append(seq)
            prompt_lens.append(tp)
            advantages.append(a)
            rewards.append(r)

    n = len(seqs)
    t_max = max(len(s) for s in seqs)
    input_ids = torch.full((n, t_max), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((n, t_max), dtype=torch.float32)
    for i, (seq, tp) in enumerate(zip(seqs, prompt_lens)):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        loss_mask[i, tp : len(seq)] = 1.0

    reward_t = torch.tensor(rewards, dtype=torch.float32)
    gen_lens = [len(seq) - tp for seq, tp in zip(seqs, prompt_lens)]
    return RolloutResult(
        batch={
            "input_ids": input_ids.to(device),
            "loss_mask": loss_mask.to(device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
        },
        reward_mean=float(reward_t.mean()),
        reward_std=float(reward_t.std(unbiased=False)),
        solve_rate=float((reward_t > 0).float().mean()),
        gen_len_mean=float(sum(gen_lens) / max(1, len(gen_lens))),
        n_samples=n,
    )
