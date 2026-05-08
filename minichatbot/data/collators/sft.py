"""SFT collator: right-pads variable-length conversations.

Right-padding is safe for a causal-attention decoder: real tokens at
positions 0..n-1 only attend to earlier positions, so they never see
the pad tokens at n..max_len-1. The pad tokens themselves do attend to
real tokens, but their loss is -100 (ignored), so no gradient flows out
of them. No attention-mask plumbing needed in the model.
"""

from __future__ import annotations

import torch

from minichatbot.data.collators import COLLATOR_REGISTRY
from minichatbot.data.collators.base import Collator


@COLLATOR_REGISTRY.register("sft")
class SFTCollator(Collator):
    """Pads `input_ids` with `pad_id` and `labels` with -100.

    Constructor takes `pad_id` directly so this collator is independent
    of any tokenizer interface. The SFT script wires it from
    `tokenizer.pad_id`.
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(s["input_ids"].size(0) for s in samples)
        B = len(samples)
        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        for i, s in enumerate(samples):
            n = s["input_ids"].size(0)
            input_ids[i, :n] = s["input_ids"]
            labels[i, :n] = s["labels"]
        return {"input_ids": input_ids, "labels": labels}
