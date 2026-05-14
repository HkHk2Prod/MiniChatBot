"""Run a fixed per-phase prompt set against a trained MiniChatBot checkpoint.

Three phases, picked via `--phase`:

    pretrain — raw next-token continuation from each prompt (no chat
        template). Output is the literal continuation appended to the
        seed.

    sft — each prompt is wrapped as a single user turn (optionally
        behind a system prompt from `benchmarks/prompts.yaml`); the
        model's assistant reply is collected.

    rl — chat-templated with the GSM8K system prompt; each completion
        is scored against its `reference` with `GSM8KReward` (1.0 iff
        the parsed final number matches). Reports per-group solve rate.

Output goes to `runs/<run_dir>/benchmark_<phase>_<UTC-timestamp>.txt`,
which keeps a permanent record next to the run that produced it — same
convention as `samples.txt`. Use `--output` to override the path.

A training callback (`type: benchmark`) calls the same engine at
end-of-run, so most users will see `benchmark_<phase>.txt` materialize
automatically; this CLI is for re-running against a different prompt
set or a different checkpoint after the fact.

Examples:
    # 100M post-RL: score GSM8K solve_rate on the latest rl_gsm8k run
    python scripts/inference/benchmark.py --phase rl --run-name rl_gsm8k

    # 100M post-SFT: probe instruction-following on the latest sft_fineweb
    python scripts/inference/benchmark.py --phase sft --run-name sft_fineweb

    # Sanity-check a pretrain checkpoint
    python scripts/inference/benchmark.py --phase pretrain --run-name pretrain_fineweb
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from minichatbot.inference.benchmark import PHASES, load_phase_prompts, run_benchmark
from minichatbot.inference.cli import (
    add_checkpoint_args,
    add_sampling_args,
    build_strategy,
    resolve_checkpoint,
    resolve_tokenizer_path,
)
from minichatbot.inference.generator import Generator
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.bpe import IM_END_TOKEN, BPETokenizer
from minichatbot.utils.torch_helpers import resolve_device


def _default_output_path(ckpt_path: Path, phase: str) -> Path:
    # ckpt_path is runs/<ts>_<name>/checkpoints/ckpt_*.pt; the run dir is
    # two levels up. Match samples.txt's convention of living alongside it.
    # Timestamp keeps multiple ad-hoc CLI runs from clobbering each other —
    # the BenchmarkCallback writes the un-suffixed `benchmark_<phase>.txt`.
    run_dir = ckpt_path.parent.parent
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return run_dir / f"benchmark_{phase}_{ts}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument(
        "--prompts-file",
        default="benchmarks/prompts.yaml",
        help="YAML with per-phase prompt sets (default: benchmarks/prompts.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .txt path. Default: runs/<run_dir>/benchmark_<phase>_<ts>.txt",
    )
    add_checkpoint_args(parser)
    # Defaults match chat.py — top_p sampling with mild penalties — so sft/rl
    # benchmarks measure what the chat REPL would actually produce. 256 is
    # generous: rl needs it for multi-step solutions; pretrain/sft simply
    # EOS-stop earlier so the cap is harmless.
    add_sampling_args(parser, default_strategy="top_p", default_max_new_tokens=256)
    parser.set_defaults(temperature=0.8)
    parser.add_argument("--frequency-penalty", type=float, default=0.5)
    parser.add_argument("--presence-penalty", type=float, default=0.5)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    try:
        phase_cfg = load_phase_prompts(args.prompts_file, args.phase)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    device = resolve_device(args.device)
    ckpt_path = resolve_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        prefer_best=True,
    )
    print(f"checkpoint: {ckpt_path}")

    tokenizer_path = resolve_tokenizer_path(ckpt_path, args.tokenizer)
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"tokenizer:  {tokenizer_path} (vocab={tokenizer.vocab_size})")

    model = LanguageModel.load(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model:      {model.cfg.type} ({n_params / 1e6:.2f}M params) on {device}")

    # sft/rl require <|im_end|> to stop at turn-end; pretrain stops at
    # the pretrain EOS (or runs to max_new_tokens if there isn't one).
    if args.phase in ("sft", "rl"):
        stop_id = tokenizer.special_token_id(IM_END_TOKEN)
        if stop_id is None:
            raise SystemExit(
                f"Phase {args.phase!r} needs a chat-tuned tokenizer with <|im_end|>. "
                f"Got a tokenizer without it — re-train it with chat specials, or "
                f"benchmark this checkpoint under --phase pretrain instead."
            )
    else:
        stop_id = tokenizer.eos_id

    strategy = build_strategy(
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    generator = Generator(
        strategy=strategy,
        eos_id=stop_id,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )

    output_path = (
        Path(args.output) if args.output else _default_output_path(ckpt_path, args.phase)
    )
    print(f"output:     {output_path}\n")

    header_lines = [
        f"checkpoint:        {ckpt_path}",
        f"strategy:          {args.strategy} (temp={args.temperature}, "
        f"top_k={args.top_k}, top_p={args.top_p})",
        f"max_new_tokens:    {args.max_new_tokens}",
        f"frequency_penalty: {args.frequency_penalty}",
        f"presence_penalty:  {args.presence_penalty}",
    ]
    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        generator=generator,
        device=device,
        phase=args.phase,
        phase_cfg=phase_cfg,
        output_path=output_path,
        max_new_tokens=args.max_new_tokens,
        header_lines=header_lines,
        verbose=True,
    )


if __name__ == "__main__":
    main()
