"""Generate completions from a trained MiniChatBot checkpoint.

Examples:
    # Auto-detect: latest checkpoint in the latest run under runs/
    python scripts/inference/generate.py "Once upon a time" "ROMEO:"

    # Filter by run_name (latest matching run)
    python scripts/inference/generate.py --run-name debug_shakespeare "ROMEO:"

    # Specific checkpoint or run dir
    python scripts/inference/generate.py --checkpoint runs/.../ckpt_step_00000200.pt "ROMEO:"
    python scripts/inference/generate.py --checkpoint runs/20260507_180000_debug_shakespeare "ROMEO:"

    # Sampling strategy + temperature
    python scripts/inference/generate.py --strategy top_p --top-p 0.9 --temperature 0.8 "..."

If `--checkpoint` is a directory, it's treated as a run dir and the latest
ckpt_step_*.pt in `{dir}/checkpoints/` is used. The tokenizer path defaults
to whatever was saved in the checkpoint's `full_config`; override with
`--tokenizer` if you've moved files around.
"""

from __future__ import annotations

import argparse

import torch

from minichatbot.inference.cli import (
    add_checkpoint_args,
    add_sampling_args,
    build_strategy,
    resolve_checkpoint,
    resolve_tokenizer_path,
)
from minichatbot.inference.generator import Generator
from minichatbot.inference.text_generator import TextGenerator
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.bpe import IM_END_TOKEN, BPETokenizer
from minichatbot.utils.torch_helpers import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate completions from a checkpoint.")
    parser.add_argument("prompts", nargs="*", help="Prompts to complete.")
    add_checkpoint_args(parser)
    add_sampling_args(parser)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    ckpt_path = resolve_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(f"checkpoint: {ckpt_path}")

    tokenizer_path = resolve_tokenizer_path(ckpt_path, args.tokenizer)
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"tokenizer:  {tokenizer_path} (vocab={tokenizer.vocab_size})")

    model = LanguageModel.load(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    n_params = sum(param.numel() for param in model.parameters())
    print(f"model:      {model.cfg.type} ({n_params / 1e6:.2f}M params) on {device}")

    strategy = build_strategy(
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

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
