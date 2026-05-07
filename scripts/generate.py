"""Generate completions from a trained MiniChatBot checkpoint.

Examples:
    # Auto-detect: latest checkpoint in the latest run under runs/
    python scripts/generate.py "Once upon a time" "ROMEO:"

    # Filter by run_name (latest matching run)
    python scripts/generate.py --run-name debug_shakespeare "ROMEO:"

    # Specific checkpoint or run dir
    python scripts/generate.py --checkpoint runs/.../ckpt_step_00000200.pt "ROMEO:"
    python scripts/generate.py --checkpoint runs/20260507_180000_debug_shakespeare "ROMEO:"

    # Sampling strategy + temperature
    python scripts/generate.py --strategy top_p --top-p 0.9 --temperature 0.8 "..."

If `--checkpoint` is a directory, it's treated as a run dir and the latest
ckpt_step_*.pt in `{dir}/checkpoints/` is used. The tokenizer path defaults
to whatever was saved in the checkpoint's `full_config`; override with
`--tokenizer` if you've moved files around.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.inference.text_generator import TextGenerator
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.bpe import IM_END_TOKEN, BPETokenizer
from minichatbot.utils.checkpoints import (
    find_latest_checkpoint,
    find_latest_checkpoint_in,
)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        p = Path(args.checkpoint)
        if p.is_file():
            return p
        if p.is_dir():
            ckpt = find_latest_checkpoint(p)
            if ckpt is None:
                raise SystemExit(f"No checkpoints found in {p}/checkpoints/")
            return ckpt
        raise SystemExit(f"Checkpoint path does not exist: {p}")

    ckpt = find_latest_checkpoint_in(args.output_dir, run_name=args.run_name)
    if ckpt is None:
        msg = f"No checkpoints found under {args.output_dir}"
        if args.run_name:
            msg += f" matching run_name='{args.run_name}'"
        msg += ". Pass --checkpoint explicitly, or train a model first."
        raise SystemExit(msg)
    return ckpt


def resolve_tokenizer_path(checkpoint_path: Path, override: str | None) -> str:
    if override is not None:
        return override
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_config = state.get("full_config")
    if full_config is None:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} has no 'full_config'. "
            "Pass --tokenizer explicitly."
        )
    path = full_config.get("tokenizer", {}).get("path")
    if path is None:
        raise SystemExit(
            f"Checkpoint at {checkpoint_path} has no tokenizer.path in its "
            "saved config. Pass --tokenizer explicitly."
        )
    return path


def build_strategy(args: argparse.Namespace):
    cls = SAMPLING_REGISTRY[args.strategy]
    if args.strategy == "greedy":
        return cls()
    if args.strategy == "temperature":
        return cls(temperature=args.temperature)
    if args.strategy == "top_k":
        return cls(k=args.top_k, temperature=args.temperature)
    if args.strategy == "top_p":
        return cls(p=args.top_p, temperature=args.temperature)
    raise SystemExit(f"Unsupported strategy: {args.strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate completions from a checkpoint.")
    parser.add_argument("prompts", nargs="*", help="Prompts to complete.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pt OR a run dir. Default: auto-detect latest under --output-dir.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json. Default: read from checkpoint's saved config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="When auto-detecting, only consider runs ending with _{run_name}.",
    )
    parser.add_argument("--output-dir", default="runs", help="Where runs live (default: runs).")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--strategy",
        default="greedy",
        choices=sorted(SAMPLING_REGISTRY.keys()),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    ckpt_path = resolve_checkpoint(args)
    print(f"checkpoint: {ckpt_path}")

    tokenizer_path = resolve_tokenizer_path(ckpt_path, args.tokenizer)
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"tokenizer:  {tokenizer_path} (vocab={tokenizer.vocab_size})")

    model = LanguageModel.load(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    n_params = sum(param.numel() for param in model.parameters())
    print(f"model:      {model.cfg.type} ({n_params / 1e6:.2f}M params) on {device}")

    strategy = build_strategy(args)

    # Prefer chat turn-end as the stop token if it exists; otherwise EOS.
    stop_id = tokenizer.special_token_id(IM_END_TOKEN)
    if stop_id is None:
        stop_id = tokenizer.eos_id

    gen = Generator(strategy=strategy, eos_id=stop_id)
    text_gen = TextGenerator(model=model, tokenizer=tokenizer, generator=gen)

    prompts = args.prompts or ["Once upon a time"]
    print(f"\nstrategy: {args.strategy} | max_new_tokens: {args.max_new_tokens}\n")

    for prompt in prompts:
        completion = text_gen.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            return_only_completion=True,
        )
        print(f"PROMPT:     {prompt}")
        print(f"COMPLETION: {completion}")
        print()


if __name__ == "__main__":
    main()
