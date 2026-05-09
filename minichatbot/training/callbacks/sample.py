"""Sample-generation callback: writes completions to {run_dir}/samples.txt."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import torch

from minichatbot.inference.generator import Generator
from minichatbot.inference.text_generator import TextGenerator
from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext
from minichatbot.utils.checkpoints import find_best_checkpoint


@CALLBACK_REGISTRY.register("sample")
class SampleGenerationCallback(Callback):
    """Generates completions for a fixed list of prompts every N steps.

    Defaults: greedy sampling (deterministic, reproducible across runs)
    and `stop_on_eos=True` — samples reflect what real generation will
    look like via `chat.py` / `generate.py`. Set `stop_on_eos: false` if
    you want to see the full `max_new_tokens` distribution mid-training
    (early samples become very short when EOS is the dominant token).

    Configure `strategy: top_k` (or `top_p`, `temperature`) and pass
    strategy kwargs alongside (e.g., `k: 50`, `temperature: 0.8`).

    Repetition control (matches `chat.py` semantics):
        frequency_penalty: subtract `f * count(token)` from each token's
            logit; defaults to 0.0 (off). Try 0.3-0.7 if greedy samples
            collapse into repeating phrases.
        presence_penalty: subtract `p` if the token has appeared at all;
            defaults to 0.0 (off).

    For SFT/chat-tuned models, set `chat_template: true`. Each prompt is
    then wrapped as a single-turn user message (`<|im_start|>user\\n...
    <|im_end|>\\n<|im_start|>assistant\\n`) before sampling, so the model
    is in its trained-on input distribution and emits an assistant reply
    instead of falling back to raw text continuation. With this enabled,
    `stop_on_eos` also stops at `<|im_end|>` (chat turn end).
    """

    def __init__(
        self,
        every: int = 500,
        prompts: list[str] | None = None,
        max_new_tokens: int = 64,
        strategy: str = "greedy",
        stop_on_eos: bool = True,
        chat_template: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **strategy_kwargs: Any,
    ) -> None:
        if every < 1:
            raise ValueError(f"SampleGenerationCallback.every must be >= 1, got {every}")
        self.every = every
        self.prompts = prompts or []
        self.max_new_tokens = max_new_tokens
        self.strategy_name = strategy
        self.stop_on_eos = stop_on_eos
        self.chat_template = chat_template
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.strategy_kwargs = strategy_kwargs
        self._text_gen: TextGenerator | None = None
        self._fh: IO[str] | None = None

    def _build_text_generator(self, ctx: CallbackContext) -> TextGenerator:
        from minichatbot.inference.strategies import SAMPLING_REGISTRY
        from minichatbot.tokenizer.bpe import IM_END_TOKEN

        strat_cls = SAMPLING_REGISTRY[self.strategy_name]
        # Default: stop on <eos> so samples mirror what `chat.py` /
        # `generate.py` will produce. Set `stop_on_eos: false` to force
        # the full max_new_tokens — useful when you want to inspect the
        # model's full distribution mid-training even after it would
        # naturally end. In chat-template mode, "stop_on_eos" stops at
        # <|im_end|> (the end-of-turn marker), not the pretrain EOS.
        eos_id: int | None = None
        if self.stop_on_eos:
            eos_id = (
                ctx.tokenizer.special_token_id(IM_END_TOKEN)
                if self.chat_template
                else ctx.tokenizer.eos_id
            )
        gen = Generator(
            strategy=strat_cls(**self.strategy_kwargs),
            eos_id=eos_id,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        return TextGenerator(model=ctx.model, tokenizer=ctx.tokenizer, generator=gen)

    def _do_generate(self) -> list[str]:
        """Run generation for the configured prompts, picking the chat-template
        path when enabled and the raw path otherwise. Returns one decoded
        completion per prompt."""
        assert self._text_gen is not None
        if self.chat_template:
            return self._text_gen.generate_chat(
                self.prompts,
                max_new_tokens=self.max_new_tokens,
                include_special_in_output=True,
            )
        return self._text_gen.generate(
            self.prompts,
            max_new_tokens=self.max_new_tokens,
            return_only_completion=True,
            include_special_in_output=True,
        )

    def on_train_start(self, ctx: CallbackContext) -> None:
        if not self.prompts or ctx.tokenizer is None:
            return
        self._text_gen = self._build_text_generator(ctx)
        path = Path(ctx.run_dir) / "samples.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._fh.write(self._config_header())
        self._fh.flush()

    def _config_header(self) -> str:
        """One-shot sampler-config block written when the file is opened.

        Surfaces the decoding settings up front so a `samples.txt` left on
        disk weeks later is still self-documenting — no need to cross-reference
        the run's `config.yaml` to know what strategy/penalties produced
        the completions.
        """
        kwargs = ", ".join(f"{k}={v}" for k, v in sorted(self.strategy_kwargs.items()))
        strategy_str = (
            f"{self.strategy_name}({kwargs})" if kwargs else self.strategy_name
        )
        return (
            "=== sampler config ===\n"
            f"strategy:          {strategy_str}\n"
            f"max_new_tokens:    {self.max_new_tokens}\n"
            f"stop_on_eos:       {self.stop_on_eos}\n"
            f"chat_template:     {self.chat_template}\n"
            f"frequency_penalty: {self.frequency_penalty}\n"
            f"presence_penalty:  {self.presence_penalty}\n"
            f"every:             {self.every}\n"
        )

    def on_step_end(self, ctx: CallbackContext) -> None:
        if self._fh is None or self._text_gen is None:
            return
        if ctx.step % self.every != 0:
            return
        self._generate(ctx)

    def on_train_end(self, ctx: CallbackContext) -> None:
        # Final block: load the best-by-val-loss checkpoint and generate
        # one last set of samples from it. The current model in memory is
        # at the latest step, which may be slightly overfit; ckpt_best.pt
        # is the model the user should compare against for SFT decisions.
        # Skipped silently if there's no eval data or no checkpoint
        # callback was configured.
        if self._fh is not None and self._text_gen is not None:
            self._generate_from_best(ctx)
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _generate_from_best(self, ctx: CallbackContext) -> None:
        assert self._fh is not None
        assert self._text_gen is not None
        best_path = find_best_checkpoint(ctx.run_dir)
        if best_path is None:
            return
        device = next(ctx.model.parameters()).device
        state = torch.load(best_path, map_location=device, weights_only=False)
        # Swap weights into the in-memory model. We're at on_train_end —
        # nothing else uses ctx.model after this, so no need to restore.
        ctx.model.load_state_dict(state["model"])
        best_step = state.get("step", "?")

        self._fh.write(f"\n=== BEST MODEL (step {best_step}) ===\n")
        completions = self._do_generate()
        for prompt, completion in zip(self.prompts, completions, strict=True):
            self._fh.write(f"PROMPT: {prompt}\n")
            self._fh.write(f"COMPLETION: {completion}\n\n")
        self._fh.flush()

    def _generate(self, ctx: CallbackContext) -> None:
        assert self._text_gen is not None
        assert self._fh is not None
        self._fh.write(f"\n=== step {ctx.step} ===\n")
        completions = self._do_generate()
        for prompt, completion in zip(self.prompts, completions, strict=True):
            self._fh.write(f"PROMPT: {prompt}\n")
            self._fh.write(f"COMPLETION: {completion}\n\n")
        self._fh.flush()
