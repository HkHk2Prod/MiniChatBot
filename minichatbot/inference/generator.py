"""Token-by-token sampling generator over a LanguageModel.

The generation loop is fixed; the per-step token-picking variance lives
in a pluggable `SamplingStrategy` (greedy / temperature / top-k / top-p),
each in its own file under `inference/strategies/`.

Repetition control (frequency_penalty, presence_penalty) lives here
rather than in the strategies because it modifies the logits before
the sampling step and applies regardless of which strategy is used —
shared concern, single implementation. OpenAI-style semantics: the
prompt counts as already-seen, penalty subtracts from logits.

If beam search or speculative decoding ever land, they should be
separate concrete classes — at that point the right shared abstraction
will be obvious. Until then, this single `Generator` class is the
whole inference layer and there's no base class to subclass against.
"""

from __future__ import annotations

import torch

from minichatbot.inference.strategies.base import SamplingStrategy
from minichatbot.inference.strategies.greedy import GreedySampling
from minichatbot.model.base import LanguageModel
from minichatbot.utils.torch_helpers import eval_mode


class Generator:
    """Token-by-token sampling generator.

    Threads the model's KV cache through `init_state` /
    `forward(state=...)`. Single-stream batched: all sequences share
    the same length budget; per-sequence EOS early-stops when every
    sequence has produced an EOS.

    Repetition penalties (OpenAI-style):
        frequency_penalty: subtract `f * count(token)` from each token's
            logit, where count includes prompt + previously generated.
            Scales with how often a token has appeared. Range typically
            [0, 2]; for small chat models 0.3-0.7 is a good starting
            point.
        presence_penalty: subtract `p` from each token's logit if it has
            appeared at all. A flat one-shot penalty regardless of count.
            Same range; combine with frequency_penalty for compounded
            effect.
        Set both to 0.0 to disable (default).
    """

    def __init__(
        self,
        strategy: SamplingStrategy | None = None,
        eos_id: int | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> None:
        self.strategy = strategy if strategy is not None else GreedySampling()
        self.eos_id = eos_id
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    @torch.no_grad()
    def generate(
        self,
        model: LanguageModel,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        with eval_mode(model):
            B = prompt_ids.size(0)
            device = prompt_ids.device
            state = model.init_state(B, device)

            out = model(prompt_ids, state=state)
            state = out.state
            generated: list[torch.Tensor] = [prompt_ids]
            last_logits = out.logits[:, -1, :]

            # Token-occurrence counts (B, V), only allocated when a
            # penalty is active. Initialized from the prompt so the
            # input counts as already-seen — matches OpenAI semantics
            # and prevents the model from immediately echoing prompt
            # tokens just because they look "fresh" from logits' view.
            counts: torch.Tensor | None = None
            penalize = self.frequency_penalty != 0.0 or self.presence_penalty != 0.0
            if penalize:
                V = last_logits.size(-1)
                counts = torch.zeros(B, V, dtype=last_logits.dtype, device=device)
                counts.scatter_add_(
                    1, prompt_ids, torch.ones_like(prompt_ids, dtype=last_logits.dtype)
                )

            # `done` is always allocated. When eos_id is None it never
            # updates (stays all-False), so done.all() is False and the
            # early-stop never fires — no Optional, no branching needed.
            done = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_new_tokens):
                if counts is not None:
                    if self.frequency_penalty != 0.0:
                        last_logits = last_logits - self.frequency_penalty * counts
                    if self.presence_penalty != 0.0:
                        last_logits = (
                            last_logits - self.presence_penalty * (counts > 0).to(last_logits.dtype)
                        )
                next_tok = self.strategy(last_logits)
                if self.eos_id is not None:
                    next_tok = torch.where(
                        done, torch.full_like(next_tok, self.eos_id), next_tok
                    )
                    done = done | (next_tok == self.eos_id)
                generated.append(next_tok.unsqueeze(1))
                if counts is not None:
                    counts.scatter_add_(
                        1,
                        next_tok.unsqueeze(1),
                        torch.ones((B, 1), dtype=counts.dtype, device=device),
                    )
                if bool(done.all()):
                    break
                out = model(next_tok.unsqueeze(1), state=state)
                state = out.state
                last_logits = out.logits[:, -1, :]
            return torch.cat(generated, dim=1)
