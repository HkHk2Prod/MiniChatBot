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
        temperature: 0.8
        top_p: 0.9                 # used when strategy == top_p
        top_k: 50                  # used when strategy == top_k
        frequency_penalty: 0.5
        presence_penalty: 0.5
        max_new_tokens: 256

`phase` is the only required param. The rest default to chat.py's flag
defaults so the benchmark output matches what an end-user chatting with
the trained model would actually see. The same `build_strategy` used by
`scripts/inference/benchmark.py` powers strategy construction, so
callback and CLI can't drift on defaults.
"""

from __future__ import annotations

from pathlib import Path

import torch

from minichatbot.inference.benchmark import PHASES, load_phase_prompts, run_benchmark
from minichatbot.inference.cli import build_strategy
from minichatbot.inference.generator import Generator
from minichatbot.rl.rewards import REWARD_REGISTRY
from minichatbot.tokenizer.bpe import IM_END_TOKEN
from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext
from minichatbot.utils.checkpoints import find_best_checkpoint
from minichatbot.utils.torch_helpers import unwrap_compiled

# Resolved relative to the repo root so the default works regardless of
# the CWD the training run was launched from — a CWD-relative default
# silently degraded any run started outside the repo root to a "skipped:
# file not found" log with no recovery path.
_DEFAULT_PROMPTS_FILE = Path(__file__).resolve().parents[3] / "benchmarks" / "prompts.yaml"


@CALLBACK_REGISTRY.register("benchmark")
class BenchmarkCallback(Callback):
    """Run the phase prompt set once at end-of-training; save to run dir."""

    def __init__(
        self,
        phase: str,
        prompts_file: str | None = None,
        max_new_tokens: int = 256,
        strategy: str = "top_p",
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(
                f"BenchmarkCallback.phase must be one of {PHASES}, got {phase!r}"
            )
        self.phase = phase
        # None -> repo-root default resolved at call time so users can still
        # opt into a CWD-relative path by passing one explicitly.
        self.prompts_file = prompts_file
        self.max_new_tokens = max_new_tokens
        self.strategy_name = strategy
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
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

        prompts_path = Path(self.prompts_file) if self.prompts_file else _DEFAULT_PROMPTS_FILE
        try:
            phase_cfg = load_phase_prompts(prompts_path, self.phase)
        except FileNotFoundError:
            print(
                f"[benchmark] skipped: prompts file {prompts_path} not "
                f"found. Pass a different `prompts_file` in the callback "
                f"params or create the file."
            )
            return

        # Fall back to the training-time system prompt when the benchmark
        # YAML doesn't pin one. Keeps the benchmark measuring what the
        # policy was actually conditioned on instead of silently drifting
        # because two files have to be edited in lockstep.
        if phase_cfg.get("system") is None and ctx.config.data.system_prompt:
            phase_cfg = {**phase_cfg, "system": ctx.config.data.system_prompt}

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

        strategy = build_strategy(
            strategy=self.strategy_name,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        generator = Generator(
            strategy=strategy,
            eos_id=stop_id,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

        # Last-step weights are what's in memory; swap in ckpt_best.pt if
        # one exists so the benchmark measures the model users will
        # actually run (best-by-val-loss, not the possibly-overfit final
        # step). Only the SFT/pretrain flows write a best checkpoint; RL
        # has no validation pass, so this is a silent no-op there.
        # `unwrap_compiled` matches save_checkpoint's symmetric path —
        # the saved state_dict has no `_orig_mod.` prefix.
        #
        # We snapshot the current weights and restore them after the run
        # because `on_train_end` fires LIFO: later callbacks (notably
        # CheckpointCallback's final periodic save) would otherwise
        # serialize the swapped-in best weights under the last-step
        # filename. Clone keeps the snapshot stable when the in-place
        # load_state_dict below overwrites the live tensors.
        device = next(ctx.model.parameters()).device
        inner = unwrap_compiled(ctx.model)
        source_label = f"final step {ctx.step}"
        best_path = find_best_checkpoint(Path(ctx.run_dir))
        saved_state: dict[str, torch.Tensor] | None = None
        if best_path is not None:
            saved_state = {k: v.detach().clone() for k, v in inner.state_dict().items()}
            state = torch.load(best_path, map_location=device, weights_only=False)
            inner.load_state_dict(state["model"])
            source_label = f"ckpt_best.pt (step {state.get('step', '?')})"

        strategy_desc = (
            f"{self.strategy_name}(temp={self.temperature}, "
            f"top_k={self.top_k}, top_p={self.top_p})"
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
        # Score the rl benchmark with the reward the run actually trained
        # on (cfg.rl.reward) so the summary measures the right objective —
        # GSM8K solve_rate for a math run, variety for a variety run. Other
        # phases don't score, so they need no reward.
        reward = (
            REWARD_REGISTRY[ctx.config.rl.reward]() if self.phase == "rl" else None
        )

        output_path = Path(ctx.run_dir) / f"benchmark_{self.phase}.txt"
        print(f"[benchmark] writing {output_path}")
        ctx.model.eval()
        try:
            run_benchmark(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                generator=generator,
                device=device,
                phase=self.phase,
                phase_cfg=phase_cfg,
                output_path=output_path,
                max_new_tokens=self.max_new_tokens,
                reward=reward,
                header_lines=header_lines,
                verbose=False,    # training stdout is already crowded
            )
        finally:
            # Restore in finally so an exception in run_benchmark — still
            # caught one frame up by on_train_end's blanket except — does
            # not leak the swapped-in weights into the rest of the
            # on_train_end LIFO chain.
            if saved_state is not None:
                inner.load_state_dict(saved_state)
