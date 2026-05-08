"""Text-in / text-out wrapper around a token-level Generator.

`Generator` is a pure tensor-in/tensor-out primitive used by token-level
consumers (RL rollouts, eval, callbacks). `TextGenerator` adds the
encode/decode layer for human-facing use cases (scripts, REPL, chat).
This split keeps the Generator abstract free of any tokenizer dependency
and avoids duplicating encode/decode logic across each sampling strategy.

Batching:
    Same-length prompts go through `Generator.generate` in a single
    forward stream. Variable-length prompts fall back to a per-prompt
    loop. True batched variable-length generation needs left-padding +
    attention masking in `LanguageModel.forward` — not yet supported.
"""

from __future__ import annotations

from typing import overload

import torch

from minichatbot.chat.template import render_prompt_for_completion
from minichatbot.inference.generator import Generator
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.base import Tokenizer


class TextGenerator:
    """Binds (model, tokenizer, generator) for text-level convenience."""

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        generator: Generator,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @overload
    def generate(
        self,
        prompts: str,
        max_new_tokens: int = ...,
        add_special: bool = ...,
        return_only_completion: bool = ...,
        skip_special: bool = ...,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = ...,
        add_special: bool = ...,
        return_only_completion: bool = ...,
        skip_special: bool = ...,
    ) -> list[str]: ...

    def generate(
        self,
        prompts: str | list[str],
        max_new_tokens: int = 64,
        add_special: bool = False,
        return_only_completion: bool = False,
        skip_special: bool = True,
    ) -> str | list[str]:
        """Generate completion text for one or many prompts (raw, no chat template).

        EOS-stopping is configured on the wrapped `Generator` at its
        construction time — `Generator(strategy, eos_id=...)`.

        Returns a `str` if `prompts` is a `str`; a `list[str]` if `prompts`
        is a list (regardless of length).

        Args:
            prompts: input string, or list of strings.
            max_new_tokens: cap on generated tokens past each prompt.
            add_special: whether to append EOS to each prompt before
                feeding the model. Defaults False.
            return_only_completion: if True, decode only the new tokens;
                if False (default), return prompt + completion concatenated.
            skip_special: if True (default), special tokens are stripped
                from the decoded output — what you want for end-user
                generation. Pass False for debug visibility (e.g. training
                samples), so collapse modes like "<eos><eos>..." are
                visible rather than hidden.
        """
        single = isinstance(prompts, str)
        prompt_list = [prompts] if single else list(prompts)
        if not prompt_list:
            return "" if single else []

        encoded = [
            self.tokenizer.encode(p, add_special=add_special) for p in prompt_list
        ]
        completions = self._generate_from_token_ids(
            encoded,
            max_new_tokens=max_new_tokens,
            return_only_completion=return_only_completion,
            skip_special=skip_special,
        )
        return completions[0] if single else completions

    @overload
    def generate_chat(
        self,
        prompts: str,
        max_new_tokens: int = ...,
        skip_special: bool = ...,
    ) -> str: ...

    @overload
    def generate_chat(
        self,
        prompts: list[str],
        max_new_tokens: int = ...,
        skip_special: bool = ...,
    ) -> list[str]: ...

    def generate_chat(
        self,
        prompts: str | list[str],
        max_new_tokens: int = 64,
        skip_special: bool = True,
    ) -> str | list[str]:
        """Generate assistant responses to one or many user prompts.

        Each prompt is wrapped as a single-turn user message via the
        ChatML chat template; an assistant header is appended; the model
        is sampled until `max_new_tokens` or the wrapped Generator's
        eos_id (typically `<|im_end|>` for chat models).

        Returns only the decoded completion (the assistant's reply), not
        the prompt prefix. Use this for SFT/chat-tuned models — for raw
        pretrained models, use `generate` instead.
        """
        single = isinstance(prompts, str)
        prompt_list = [prompts] if single else list(prompts)
        if not prompt_list:
            return "" if single else []

        encoded = [
            render_prompt_for_completion(
                [{"role": "user", "content": p}], self.tokenizer
            )
            for p in prompt_list
        ]
        completions = self._generate_from_token_ids(
            encoded,
            max_new_tokens=max_new_tokens,
            return_only_completion=True,
            skip_special=skip_special,
        )
        return completions[0] if single else completions

    def _generate_from_token_ids(
        self,
        encoded: list[list[int]],
        max_new_tokens: int,
        return_only_completion: bool,
        skip_special: bool,
    ) -> list[str]:
        """Shared core: token-ids in, decoded strings out."""
        if len({len(ids) for ids in encoded}) == 1:
            # Same length across all prompts — one batched forward stream.
            prompt_ids = torch.tensor(encoded, dtype=torch.long, device=self.device)
            out = self.generator.generate(
                self.model, prompt_ids, max_new_tokens=max_new_tokens
            )
            out_ids_per_prompt = [out[i].tolist() for i in range(out.size(0))]
        else:
            # Variable length — fall back to per-prompt iteration.
            out_ids_per_prompt = []
            for ids in encoded:
                prompt_ids = torch.tensor(
                    [ids], dtype=torch.long, device=self.device
                )
                out = self.generator.generate(
                    self.model, prompt_ids, max_new_tokens=max_new_tokens
                )
                out_ids_per_prompt.append(out[0].tolist())

        completions: list[str] = []
        for prompt_ids_list, full_ids in zip(
            encoded, out_ids_per_prompt, strict=True
        ):
            decode_ids = (
                full_ids[len(prompt_ids_list):] if return_only_completion else full_ids
            )
            completions.append(
                self.tokenizer.decode(decode_ids, skip_special=skip_special)
            )
        return completions
