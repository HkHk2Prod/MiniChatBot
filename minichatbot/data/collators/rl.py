"""RL prompt collator: groups prompts without padding them.

The RL step replicates each prompt `group_size` times and samples a
completion group, then re-batches the (prompt + completion) sequences
itself (see `minichatbot.rl.collect_rollouts`). So there's nothing to
gain from padding prompts together here — and `reference` is a string,
not a tensor. This collator therefore just transposes the list of
sample dicts into a dict of lists:

    [{"prompt_ids": t0, "reference": r0}, {"prompt_ids": t1, ...}, ...]
        -> {"prompt_ids": [t0, t1, ...], "reference": [r0, r1, ...]}

`GRPOTrainer._train_step` consumes these lists directly.
"""

from __future__ import annotations

from typing import Any

import torch

from minichatbot.data.collators import COLLATOR_REGISTRY
from minichatbot.data.collators.base import Collator


@COLLATOR_REGISTRY.register("rl")
class RLCollator(Collator):
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_ids: list[torch.Tensor] = [s["prompt_ids"] for s in samples]
        references: list[str] = [s["reference"] for s in samples]
        return {"prompt_ids": prompt_ids, "reference": references}
