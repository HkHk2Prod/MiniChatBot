"""Adapter exposing a MiniChatBot model to EleutherAI's lm-evaluation-harness.

The harness drives any model through three methods on `lm_eval.api.model.LM`:

    loglikelihood          (context, continuation) -> (logprob, is_greedy)
        The workhorse for multiple-choice tasks (piqa, arc, hellaswag,
        lambada accuracy, ...): score each candidate, pick the argmax.
    loglikelihood_rolling  (string,) -> logprob
        Whole-document loglikelihood for perplexity tasks (wikitext,
        lambada_openai ppl), computed over rolling windows of the context.
    generate_until         (context, gen_kwargs) -> text
        Free generation up to stop strings, for tasks like gsm8k.

`loglikelihood` / `loglikelihood_rolling` delegate to the dependency-free
`scoring.score_batch`; `generate_until` wraps the existing `Generator`.
This module DOES require `lm-eval` (it subclasses the harness's `LM`); the
scoring core does not, so unit tests can exercise the math without it.

Usage (see scripts/inference/eval_harness.py):

    import lm_eval
    adapter = MiniChatBotLM(model, tokenizer, device=device)
    results = lm_eval.simple_evaluate(model=adapter, tasks=["piqa", "lambada_openai"])
"""

from __future__ import annotations

from typing import Any

import torch
from lm_eval.api.model import LM

from minichatbot.eval.scoring import score_batch
from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies.greedy import GreedySampling
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.base import Tokenizer


# Not decorated with @register_model: we pass the adapter instance straight to
# `lm_eval.simple_evaluate(model=...)`, so it never needs to be looked up by
# string from lm-eval's CLI. (The decorator also erases the constructor
# signature for type checkers, flagging every instantiation.)
class MiniChatBotLM(LM):
    """Wrap a `LanguageModel` + `Tokenizer` as an lm-eval `LM`.

    Pass an instance straight to `lm_eval.simple_evaluate(model=...)`. The
    model is used in eval mode under `torch.no_grad`; scoring runs in fp32
    for deterministic, comparable numbers regardless of training precision.
    """

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        *,
        device: torch.device,
        batch_size: int = 8,
        max_gen_toks: int = 256,
        eos_id: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = model.cfg.max_seq_len
        # Conditioning token used when a context is empty, and as the prefix
        # for rolling-window perplexity. Prefer BOS, fall back to EOS.
        self._prefix_id = tokenizer.bos_id if tokenizer.bos_id is not None else tokenizer.eos_id
        # Generation stops at this token (caller picks: pretrain EOS, or the
        # chat <|im_end|> for instruction-tuned checkpoints).
        gen_eos = eos_id if eos_id is not None else tokenizer.eos_id
        self.generator = Generator(strategy=GreedySampling(), eos_id=gen_eos)

    # --- properties the harness reads -------------------------------------

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def eot_token_id(self) -> int:
        return self.tok.eos_id

    # --- tokenization helpers ---------------------------------------------

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Tokenize (context, continuation) as a *whole* then split.

        Encoding the two separately can differ from encoding the joined
        string because BPE may merge across the boundary; we tokenize
        `context + continuation` and slice off the context's token count so
        the continuation tokens are exactly those the model must predict.
        Trailing whitespace on the context is moved onto the continuation
        first (lm-eval convention) so the split lands on a token boundary.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tok.encode(context + continuation, include_special=False)
        ctx_enc = self.tok.encode(context, include_special=False) if context else []
        return ctx_enc, whole[len(ctx_enc):]

    def _score_pairs(self, pairs: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        """Score token-id pairs, batched. Sorts by length for padding
        efficiency, then restores the caller's order."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out: list[tuple[float, bool] | None] = [None] * len(pairs)
        for start in range(0, len(order), self._batch_size):
            idx = order[start : start + self._batch_size]
            scored = score_batch(
                self.model,
                [pairs[i] for i in idx],
                device=self._device,
                max_length=self._max_length,
                prefix_id=self._prefix_id,
            )
            for i, res in zip(idx, scored, strict=True):
                out[i] = res
        return [r for r in out if r is not None]

    # --- LM interface ------------------------------------------------------

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        pairs = [self._encode_pair(*req.args) for req in requests]
        return self._score_pairs(pairs)

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        # Roll the document through fixed-size windows (each token predicted
        # from up to max_length-1 preceding tokens) and sum the per-window
        # loglikelihoods. lm-eval ships the canonical windowing helper.
        from lm_eval.utils import get_rolling_token_windows

        results: list[float] = []
        for req in requests:
            (string,) = req.args
            token_ids = self.tok.encode(string, include_special=False)
            windows = list(
                get_rolling_token_windows(
                    token_list=token_ids,
                    prefix_token=self._prefix_id,
                    max_seq_len=self._max_length,
                    context_len=1,
                )
            )
            scored = self._score_pairs([(ctx, cont) for ctx, cont in windows])
            results.append(sum(lp for lp, _ in scored))
        return results

    def generate_until(self, requests: list[Any]) -> list[str]:
        results: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until") or []
            max_gen = int(gen_kwargs.get("max_gen_toks", self._max_gen_toks))
            # Reserve room for the generated tokens inside the context window.
            keep = max(self._max_length - max_gen, 1)
            ctx_ids = self.tok.encode(context, include_special=False)[-keep:]
            prompt = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)
            out = self.generator.generate(self.model, prompt, max_new_tokens=max_gen)
            text = self.tok.decode(out[0].tolist()[len(ctx_ids):], include_special=False)
            for stop in until:
                if stop:
                    text = text.split(stop)[0]
            results.append(text)
        return results
