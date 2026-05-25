"""Loglikelihood scoring core — the dependency-free heart of the lm-eval glue.

Given (context, continuation) token-id pairs, return the model's summed
log-probability of the continuation tokens plus whether the model would
have greedily produced them. This is the single primitive behind every
loglikelihood-based task in EleutherAI's harness (multiple-choice tasks
score each candidate this way; perplexity tasks roll it over windows).

No `lm-eval` dependency lives here on purpose: the math is testable on a
freshly-built model without pulling in the harness. `lm_eval_adapter.py`
imports this and exposes it through the harness's `LM` interface.

Batching/padding notes
----------------------
We right-pad each batch to its longest sequence and run a single forward.
This is safe *without* an attention mask because the model is causal:
position `p` attends only to positions `<= p`, so a real token never sees
a trailing pad, and RoPE positions of the real (front-loaded) tokens are
unchanged. We only ever read logits at real continuation positions, so the
garbage logits over pad positions are simply ignored. Run in fp32
(`logits.float()`) for deterministic, comparable scores.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minichatbot.model.base import LanguageModel
from minichatbot.utils.torch_helpers import eval_mode


def _prepare(
    context_ids: list[int],
    continuation_ids: list[int],
    *,
    max_length: int,
    prefix_id: int,
) -> tuple[list[int], int]:
    """Build the model input and count how many trailing tokens are scored.

    Returns `(input_ids, n_cont)` where `input_ids` is `context + continuation`
    left-truncated to `max_length` (continuation is kept; oldest context
    tokens are dropped first — the harness convention), and `n_cont` is the
    number of continuation tokens actually scored. A continuation needs at
    least one preceding token to be predicted from, so `n_cont` is capped at
    `len(input_ids) - 1`; when the context is empty we prepend `prefix_id`
    (bos/eos) to supply that conditioning token.
    """
    if not context_ids:
        context_ids = [prefix_id]
    inp = context_ids + continuation_ids
    if len(inp) > max_length:
        inp = inp[-max_length:]
    n_cont = min(len(continuation_ids), len(inp) - 1)
    return inp, max(n_cont, 0)


@torch.no_grad()
def score_batch(
    model: LanguageModel,
    batch: list[tuple[list[int], list[int]]],
    *,
    device: torch.device,
    max_length: int,
    prefix_id: int,
) -> list[tuple[float, bool]]:
    """Score one batch of (context_ids, continuation_ids) pairs.

    Returns one `(logprob_sum, is_greedy)` per input pair, in the same order:
      - `logprob_sum`: summed log P(continuation token | preceding tokens).
      - `is_greedy`:   True iff every continuation token was the argmax — i.e.
                       the model would have produced this continuation greedily.
    A degenerate empty continuation (n_cont == 0) scores `(0.0, True)`.
    """
    prepared = [
        _prepare(ctx, cont, max_length=max_length, prefix_id=prefix_id)
        for ctx, cont in batch
    ]
    max_t = max(len(inp) for inp, _ in prepared)
    bsz = len(prepared)

    # Right-pad with 0; pad positions are never read (see module docstring).
    input_ids = torch.zeros(bsz, max_t, dtype=torch.long, device=device)
    for i, (inp, _) in enumerate(prepared):
        input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long, device=device)

    with eval_mode(model):
        logits = model(input_ids).logits  # (B, T, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    results: list[tuple[float, bool]] = []
    for i, (inp, n_cont) in enumerate(prepared):
        if n_cont == 0:
            results.append((0.0, True))
            continue
        length = len(inp)
        # Continuation occupies inp[length - n_cont : length]; token at
        # position p is predicted by the logits at position p - 1.
        targets = input_ids[i, length - n_cont : length]              # (n_cont,)
        pred = log_probs[i, length - n_cont - 1 : length - 1]          # (n_cont, V)
        token_lp = pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (n_cont,)
        is_greedy = bool((pred.argmax(dim=-1) == targets).all().item())
        results.append((float(token_lp.sum().item()), is_greedy))
    return results
