"""Multiple-choice collator: flattens a batch of MC examples into rows.

Each example carries N candidate sequences (`context + choice`). This
collator concatenates the candidates of every example in the batch into a
single flat set of rows (so one forward pass scores them all), right-pads
them to a common length, and records the bookkeeping the DPO loss needs:

    input_ids    (R, T)  every candidate, right-padded with pad_id
    lengths      (R,)    real (pre-pad) length of each row
    n_conts      (R,)    scored continuation tokens per row
    cont_chars   (R,)    continuation character length per row (char norm)
    group_index  (R,)    which example (0..B-1) each row belongs to
    is_gold      (R,)    1.0 on each example's gold candidate, else 0.0

Right-padding is safe for a causal decoder (real tokens never attend to
trailing pads), and the loss only reads logits at real continuation
positions — same reasoning as the lm-eval scoring core.
"""

from __future__ import annotations

from typing import Any

import torch

from minichatbot.data.collators import COLLATOR_REGISTRY
from minichatbot.data.collators.base import Collator
from minichatbot.tokenizer.base import Tokenizer


@COLLATOR_REGISTRY.register("mc")
class MCCollator(Collator):
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    @classmethod
    def from_config(cls, tokenizer: Tokenizer) -> MCCollator:
        return cls(pad_id=tokenizer.pad_id)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        n_conts: list[int] = []
        cont_chars: list[int] = []
        group_index: list[int] = []
        is_gold: list[float] = []

        for g, s in enumerate(samples):
            gold = int(s["gold"])
            for j, row in enumerate(s["input_rows"]):
                rows.append(row)
                lengths.append(int(row.size(0)))
                n_conts.append(int(s["n_conts"][j]))
                cont_chars.append(int(s["cont_chars"][j]))
                group_index.append(g)
                is_gold.append(1.0 if j == gold else 0.0)

        max_len = max(lengths)
        input_ids = torch.full((len(rows), max_len), self.pad_id, dtype=torch.long)
        for r, row in enumerate(rows):
            input_ids[r, : lengths[r]] = row

        return {
            "input_ids": input_ids,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "n_conts": torch.tensor(n_conts, dtype=torch.long),
            "cont_chars": torch.tensor(cont_chars, dtype=torch.long),
            "group_index": torch.tensor(group_index, dtype=torch.long),
            "is_gold": torch.tensor(is_gold, dtype=torch.float),
        }
