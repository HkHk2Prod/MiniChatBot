"""Token-by-token sampling generator over a LanguageModel.

The generation loop is fixed; the per-step token-picking variance lives
in a pluggable `SamplingStrategy` (greedy / temperature / top-k / top-p),
each in its own file under `inference/strategies/`.

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
    """

    def __init__(
        self,
        strategy: SamplingStrategy | None = None,
        eos_id: int | None = None,
    ) -> None:
        self.strategy = strategy if strategy is not None else GreedySampling()
        self.eos_id = eos_id

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

            # `done` is always allocated. When eos_id is None it never
            # updates (stays all-False), so done.all() is False and the
            # early-stop never fires — no Optional, no branching needed.
            done = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_new_tokens):
                next_tok = self.strategy(last_logits)
                if self.eos_id is not None:
                    next_tok = torch.where(
                        done, torch.full_like(next_tok, self.eos_id), next_tok
                    )
                    done = done | (next_tok == self.eos_id)
                generated.append(next_tok.unsqueeze(1))
                if bool(done.all()):
                    break
                out = model(next_tok.unsqueeze(1), state=state)
                state = out.state
                last_logits = out.logits[:, -1, :]
            return torch.cat(generated, dim=1)
