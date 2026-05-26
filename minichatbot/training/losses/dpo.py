"""DPO loss for multiple-choice preference optimization.

Operates on a batch of flattened candidate rows (see the `mc` collator)
plus the reference model's per-row log-probabilities, which the
`DPOTrainer` precomputes and stashes in `batch["ref_logp"]` (analogous to
GRPO precomputing advantages in its rollout).

Per candidate row the implicit reward is the reference-anchored log-ratio

    rᵢ = β · ( logπθ(yᵢ | x) − logπ_ref(yᵢ | x) ) / Zᵢ

(optionally length-normalized by token count or continuation chars, Zᵢ).
Within each example we then take the softmax over its candidates and a
cross-entropy against the gold index — the Plackett-Luce / N-way
generalization of DPO. For exactly two candidates this reduces to the
standard pairwise DPO logistic on the log-ratio spread.

`logπθ` flows gradients; `logπ_ref` is a constant. Pushing the gold
candidate's log-ratio above the distractors' is exactly what makes the
benchmark's argmax over policy scores land on gold.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minichatbot.model.base import ModelOutput
from minichatbot.training.losses import LOSS_REGISTRY
from minichatbot.training.losses.base import Loss

_VALID_NORMS = {"none", "token", "char"}


def sequence_logp(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    lengths: torch.Tensor,
    n_conts: torch.Tensor,
) -> torch.Tensor:
    """Summed log P(continuation token | preceding tokens) per row.

    Mirrors `minichatbot.eval.scoring.score_batch`, but differentiable: for
    row r with real length L and n_cont c, the continuation occupies
    `input_ids[r, L-c:L]` and is predicted by `logits[r, L-c-1:L-1]`.
    log-softmax is taken in fp32 over just the scored slice (cheap, and
    matches the eval's fp32 scoring) — never over the full (R, T, V) tensor.
    Returns a (R,) tensor.
    """
    out = []
    for r in range(logits.size(0)):
        length = int(lengths[r])
        c = int(n_conts[r])
        if c <= 0:
            out.append(logits.new_zeros(()))
            continue
        targets = input_ids[r, length - c : length]              # (c,)
        pred = logits[r, length - c - 1 : length - 1].float()    # (c, V)
        tok_lp = pred.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out.append(tok_lp.sum())
    return torch.stack(out)


@LOSS_REGISTRY.register("dpo")
class DPOLoss(Loss):
    """N-way reference-anchored preference loss. Reads `batch["ref_logp"]`.

    `beta` scales the implicit reward; `score_norm` ('none' | 'token' |
    'char') length-normalizes the per-row log-prob so the optimized score
    matches the benchmark's metric (raw `acc` vs length-normalized
    `acc_norm`). After each forward, `last_acc` / `last_margin` hold the
    final group's argmax accuracy and gold-vs-best-distractor score margin
    for monitoring.
    """

    def __init__(self, beta: float = 0.1, score_norm: str = "none") -> None:
        super().__init__()
        if score_norm not in _VALID_NORMS:
            raise ValueError(f"score_norm={score_norm!r}; expected one of {sorted(_VALID_NORMS)}")
        self.beta = beta
        self.score_norm = score_norm
        self.last_acc: float = 0.0
        self.last_margin: float = 0.0

    def _denom(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.score_norm == "token":
            return batch["n_conts"].clamp(min=1).float()
        if self.score_norm == "char":
            return batch["cont_chars"].clamp(min=1).float()
        return torch.ones(batch["input_ids"].size(0), device=batch["input_ids"].device)

    def forward(
        self,
        output: ModelOutput,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pol = sequence_logp(output.logits, batch["input_ids"], batch["lengths"], batch["n_conts"])
        ref = batch["ref_logp"]
        denom = self._denom(batch)
        # The metric-matched per-candidate score, and the DPO log-ratio reward.
        score = pol / denom
        reward = self.beta * (score - ref / denom)

        group_index = batch["group_index"]
        is_gold = batch["is_gold"]
        n_groups = int(group_index.max().item()) + 1

        losses = []
        accs = []
        margins = []
        for g in range(n_groups):
            mask = group_index == g
            reward_g = reward[mask]
            gold_pos = int(is_gold[mask].argmax().item())
            target = torch.tensor([gold_pos], device=reward_g.device)
            losses.append(F.cross_entropy(reward_g.unsqueeze(0), target))
            with torch.no_grad():
                score_g = score[mask]
                accs.append(1.0 if int(score_g.argmax().item()) == gold_pos else 0.0)
                others = torch.cat([score_g[:gold_pos], score_g[gold_pos + 1 :]])
                margins.append(
                    float((score_g[gold_pos] - others.max()).item()) if others.numel() else 0.0
                )

        self.last_acc = sum(accs) / len(accs)
        self.last_margin = sum(margins) / len(margins)
        return torch.stack(losses).mean()
