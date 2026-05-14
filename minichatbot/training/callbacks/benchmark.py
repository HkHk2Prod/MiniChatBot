"""End-of-training benchmark callback: writes `benchmark_<phase>.txt`.

At `on_train_end`, runs the phase's near/generalize/ood prompt sets
(from `benchmarks/prompts.yaml` by default) through the final model and
saves the output alongside checkpoints. Mirrors what
`scripts/inference/benchmark.py` does, but firing automatically from
inside every training run so each run dir is self-documenting.

If a `ckpt_best.pt` exists at end of training (only the SFT/pretrain
flows produce one — RL has no validation pass), the callback swaps the
best-by-val-loss weights into the model before running the benchmark.
The in-memory model is at the last step, which may be slightly
overfit; the best-by-val checkpoint is what users actually deploy, so
that's what the benchmark should measure.

Configure in your run's `callbacks:` block:

    - type: benchmark
      params:
        phase: rl                  # pretrain | sft | rl
        strategy: top_p            # any registered SamplingStrategy
        p: 0.9                     # strategy kwargs (e.g. p / k / temperature)
        temperature: 0.8
        frequency_penalty: 0.5
        presence_penalty: 0.5
        max_new_tokens: 256

`phase` is the only required param. The rest default to chat.py's flag
defaults so the benchmark output matches what an end-user chatting with
the trained model would actually see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from minichatbot.inference.benchmark import PHASES, load_phase_prompts, run_benchmark
from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.tokenizer.bpe import IM_END_TOKEN
from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext
from minichatbot.utils.checkpoints import find_best_checkpoint


@CALLBACK_REGISTRY.register("benchmark")
class BenchmarkCallback(Callback):
    """Run the phase prompt set once at end-of-training; save to run dir."""

    def __init__(
        self,
        phase: str,
        prompts_file: str = "benchmarks/prompts.yaml",
        max_new_tokens: int = 256,
        strategy: str = "top_p",
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        **strategy_kwargs: Any,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(
                f"BenchmarkCallback.phase must be one of {PHASES}, got {phase!r}"
            )
        # Sane chat-style defaults for the two pluggable strategy kwargs
        # so callers can omit them. Explicit kwargs still win.
        if strategy == "top_p":
            strategy_kwargs.setdefault("p", 0.9)
            strategy_kwargs.setdefault("temperature", 0.8)
        elif strategy == "top_k":
            strategy_kwargs.setdefault("k", 50)
            strategy_kwargs.setdefault("temperature", 0.8)
        elif strategy == "temperature":
            strategy_kwargs.setdefault("temperature", 0.8)
        self.phase = phase
        self.prompts_file = prompts_file
        self.max_new_tokens = max_new_tokens
        self.strategy_name = strategy
        self.strategy_kwargs = strategy_kwargs
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def on_train_end(self, ctx: CallbackContext) -> None:
        # Defensive: training callbacks are observers; a benchmark failure
        # MUST NOT crash a run that otherwise succeeded. Wrap the whole
        # body in a try/except and log instead of re-raising.
        try:
            self._run(ctx)
        except Exception as exc:                                # noqa: BLE001
            print(f"[benchmark] skipped: {exc}")

    def _run(self, ctx: CallbackContext) -> None:
        if ctx.tokenizer is None:
            print("[benchmark] skipped: trainer didn't provide a tokenizer.")
            return

        try:
            phase_cfg = load_phase_prompts(self.prompts_file, self.phase)
        except FileNotFoundError:
            print(
                f"[benchmark] skipped: prompts file {self.prompts_file!r} not "
                f"found. Pass a different `prompts_file` in the callback "
                f"params or create the file."
            )
            return

        # Chat phases need the chat turn-end; pretrain stops at pretrain EOS.
        if self.phase in ("sft", "rl"):
            stop_id = ctx.tokenizer.special_token_id(IM_END_TOKEN)
            if stop_id is None:
                print(
                    f"[benchmark] skipped: phase {self.phase!r} needs a "
                    f"tokenizer with <|im_end|>; this run's tokenizer "
                    f"doesn't have one."
                )
                return
        else:
            stop_id = ctx.tokenizer.eos_id

        strat_cls = SAMPLING_REGISTRY[self.strategy_name]
        generator = Generator(
            strategy=strat_cls(**self.strategy_kwargs),
            eos_id=stop_id,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

        # Last-step weights are what's in memory; swap in ckpt_best.pt if
        # one exists so the benchmark measures the model users will
        # actually run (best-by-val-loss, not the possibly-overfit final
        # step). Only the SFT/pretrain flows write a best checkpoint; RL
        # has no validation pass, so this is a silent no-op there.
        device = next(ctx.model.parameters()).device
        source_label = f"final step {ctx.step}"
        best_path = find_best_checkpoint(Path(ctx.run_dir))
        if best_path is not None:
            state = torch.load(best_path, map_location=device, weights_only=False)
            ctx.model.load_state_dict(state["model"])
            source_label = f"ckpt_best.pt (step {state.get('step', '?')})"

        kwarg_str = ", ".join(f"{k}={v}" for k, v in sorted(self.strategy_kwargs.items()))
        strategy_desc = (
            f"{self.strategy_name}({kwarg_str})" if kwarg_str else self.strategy_name
        )
        header_lines = [
            f"run_dir:           {ctx.run_dir}",
            f"weights:           {source_label}",
            f"strategy:          {strategy_desc}",
            f"max_new_tokens:    {self.max_new_tokens}",
            f"frequency_penalty: {self.frequency_penalty}",
            f"presence_penalty:  {self.presence_penalty}",
        ]

        # Stable filename (no timestamp): one benchmark per training run,
        # written next to checkpoints — anyone scanning the run dir gets
        # an obvious "what does this model do?" view. The CLI script
        # appends a timestamp for ad-hoc re-runs that shouldn't clobber.
        output_path = Path(ctx.run_dir) / f"benchmark_{self.phase}.txt"
        print(f"[benchmark] writing {output_path}")
        ctx.model.eval()
        run_benchmark(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            generator=generator,
            device=device,
            phase=self.phase,
            phase_cfg=phase_cfg,
            output_path=output_path,
            max_new_tokens=self.max_new_tokens,
            header_lines=header_lines,
            verbose=False,    # training stdout is already crowded
        )
