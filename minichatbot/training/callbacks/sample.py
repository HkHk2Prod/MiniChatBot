"""Sample-generation callback: writes completions to {run_dir}/samples.txt."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

from minichatbot.inference.generator import Generator
from minichatbot.inference.text_generator import TextGenerator
from minichatbot.tokenizer.bpe import (
    BOS_TOKEN,
    EOS_TOKEN,
    IM_END_TOKEN,
    IM_START_TOKEN,
    PAD_TOKEN,
    RESERVED_TOKENS,
)
from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext

# All known special tokens, longest-first so e.g. "<|im_start|>" is matched
# before any partial substring overlap. Sorted len-desc avoids edge cases
# where one token is a prefix of another.
_SPECIAL_TOKENS: list[str] = sorted(
    [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, IM_START_TOKEN, IM_END_TOKEN, *RESERVED_TOKENS],
    key=len,
    reverse=True,
)

_ANSI_YELLOW = "\x1b[33m"
_ANSI_RESET = "\x1b[0m"


def _colorize_specials(text: str) -> str:
    """Wrap each special-token occurrence in ANSI yellow.

    Renders as colored text in any ANSI-aware viewer (terminals, `cat`/`less`,
    VS Code's output panel). Plain text editors will show literal escape
    sequences — disable via `colorize: false` if that's your primary view.
    """
    for tok in _SPECIAL_TOKENS:
        if tok in text:
            text = text.replace(tok, f"{_ANSI_YELLOW}{tok}{_ANSI_RESET}")
    return text


@CALLBACK_REGISTRY.register("sample")
class SampleGenerationCallback(Callback):
    """Generates completions for a fixed list of prompts every N steps.

    Defaults: greedy sampling (deterministic, reproducible across runs),
    `stop_on_eos=False` (always emit the full `max_new_tokens`), and
    `colorize=True` (ANSI-yellow special tokens for visibility in terminals
    and ANSI-aware viewers).

    The no-stop default exists because undertrained models predict `<eos>`
    very early — stopping there hides what the model is actually learning
    and makes early samples look empty. Pass `stop_on_eos: true` once the
    model is mature enough that EOS-stopping reflects real completion.

    Configure `strategy: top_k` (or `top_p`, `temperature`) and pass
    strategy kwargs alongside (e.g., `k: 50`, `temperature: 0.8`).

    Disable `colorize` if you primarily view samples.txt in a plain text
    editor (escape sequences will appear as literal characters there).
    """

    def __init__(
        self,
        every: int = 500,
        prompts: list[str] | None = None,
        max_new_tokens: int = 64,
        strategy: str = "greedy",
        stop_on_eos: bool = False,
        colorize: bool = True,
        **strategy_kwargs: Any,
    ) -> None:
        if every < 1:
            raise ValueError(f"SampleGenerationCallback.every must be >= 1, got {every}")
        self.every = every
        self.prompts = prompts or []
        self.max_new_tokens = max_new_tokens
        self.strategy_name = strategy
        self.stop_on_eos = stop_on_eos
        self.colorize = colorize
        self.strategy_kwargs = strategy_kwargs
        self._text_gen: TextGenerator | None = None
        self._fh: IO[str] | None = None

    def _build_text_generator(self, ctx: CallbackContext) -> TextGenerator:
        from minichatbot.inference.strategies import SAMPLING_REGISTRY
        strat_cls = SAMPLING_REGISTRY[self.strategy_name]
        # Default: do NOT stop on <eos> for training samples. An undertrained
        # model collapses to predicting <eos> within a few tokens (it's one of
        # the most frequent tokens in the corpus), making "completion = single
        # token" look like the model learned nothing — when really it just
        # short-circuited. Forcing the full max_new_tokens shows the actual
        # token distribution the model has converged to.
        eos_id = ctx.tokenizer.eos_id if self.stop_on_eos else None
        gen = Generator(
            strategy=strat_cls(**self.strategy_kwargs),
            eos_id=eos_id,
        )
        return TextGenerator(model=ctx.model, tokenizer=ctx.tokenizer, generator=gen)

    def on_train_start(self, ctx: CallbackContext) -> None:
        if not self.prompts or ctx.tokenizer is None:
            return
        self._text_gen = self._build_text_generator(ctx)
        path = Path(ctx.run_dir) / "samples.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def on_step_end(self, ctx: CallbackContext) -> None:
        if self._fh is None or self._text_gen is None:
            return
        if ctx.step % self.every != 0:
            return
        self._generate(ctx)

    def on_train_end(self, ctx: CallbackContext) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _generate(self, ctx: CallbackContext) -> None:
        assert self._text_gen is not None
        assert self._fh is not None
        self._fh.write(f"\n=== step {ctx.step} ===\n")
        completions = self._text_gen.generate(
            self.prompts,
            max_new_tokens=self.max_new_tokens,
            return_only_completion=True,
            skip_special=False,
        )
        for prompt, completion in zip(self.prompts, completions, strict=True):
            if self.colorize:
                completion = _colorize_specials(completion)
            self._fh.write(f"PROMPT: {prompt}\n")
            self._fh.write(f"COMPLETION: {completion}\n\n")
        self._fh.flush()
