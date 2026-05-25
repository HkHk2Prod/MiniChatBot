"""Run EleutherAI's lm-evaluation-harness against a trained MiniChatBot checkpoint.

Evaluates the model on standardized tasks (lambada, piqa, arc, hellaswag,
wikitext perplexity, gsm8k, ...) for numbers comparable to published
models — complementing the curated, human-readable prompt sets that
`scripts/inference/benchmark.py` produces.

Requires the optional `eval` extra:
    pip install -e ".[eval]"

Examples:
    # Base model: language-modeling + commonsense MC (the tasks with real
    # signal at this scale). 0-shot to fit the short context window.
    python scripts/inference/eval_harness.py --run-name pretrain_fineweb \
        --tasks lambada_openai,piqa,arc_easy,hellaswag,wikitext --num-fewshot 0

    # RL checkpoint: score the GSM8K objective with the standard harness.
    python scripts/inference/eval_harness.py --run-name rl_gsm8k --tasks gsm8k --chat

    # Quick smoke pass: cap examples per task.
    python scripts/inference/eval_harness.py --run-name sft_fineweb --tasks piqa --limit 50

Named eval_harness.py rather than lm_eval.py on purpose: running a file puts
its own directory on sys.path[0], so a script called lm_eval.py would shadow
the installed `lm_eval` package it imports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from minichatbot.inference.cli import (
    add_checkpoint_args,
    resolve_checkpoint,
    resolve_tokenizer_path,
)
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.bpe import IM_END_TOKEN, BPETokenizer
from minichatbot.utils.torch_helpers import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Run lm-evaluation-harness on a checkpoint.").splitlines()[0]
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated lm-eval task names (e.g. lambada_openai,piqa,gsm8k).",
    )
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot examples (default 0).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap examples per task for a fast pass (default: full task).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Loglikelihood batch size.")
    parser.add_argument(
        "--max-gen-toks", type=int, default=256, help="Max tokens for generative tasks."
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Instruction-tuned checkpoint: stop generation at <|im_end|> instead of pretrain EOS.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Results .json path. Default: runs/<run_dir>/lm_eval_<ts>.json",
    )
    add_checkpoint_args(parser)
    args = parser.parse_args()

    # Import here so a missing optional dependency is a clean message, not an
    # ImportError traceback at module load.
    try:
        import lm_eval
        from lm_eval.utils import make_table
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "lm-evaluation-harness is not installed. Install the optional extra:\n"
            '    pip install -e ".[eval]"'
        ) from exc

    from minichatbot.eval.lm_eval_adapter import MiniChatBotLM

    if args.seed is not None:
        torch.manual_seed(args.seed)

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
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model:      {model.cfg.type} ({n_params / 1e6:.2f}M params) on {device}")

    # Chat checkpoints stop generation at the turn-end token; base models stop
    # at the pretrain EOS. Only matters for generative tasks (e.g. gsm8k).
    eos_id = None
    if args.chat:
        eos_id = tokenizer.special_token_id(IM_END_TOKEN)
        if eos_id is None:
            raise SystemExit(
                "--chat needs a tokenizer with <|im_end|>; this checkpoint's "
                "tokenizer doesn't have it. Drop --chat for a base model."
            )

    adapter = MiniChatBotLM(
        model,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_gen_toks=args.max_gen_toks,
        eos_id=eos_id,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"tasks:      {tasks} (num_fewshot={args.num_fewshot}, limit={args.limit})\n")

    # lm-eval's `simple_evaluate` is dynamically typed (it's decorated, so the
    # checker can't see its real signature) and returns an untyped dict. Route
    # it through `Any` so the type checker doesn't flag the real kwargs or the
    # `results.get(...)` below — runtime is covered by the smoke tests.
    evaluate_fn: Any = lm_eval.simple_evaluate
    results = evaluate_fn(
        model=adapter,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    print(make_table(results))

    run_dir = ckpt_path.parent.parent
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_path = run_dir / f"lm_eval_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # `results` carries non-JSON bits (e.g. task configs); persist just the
    # scores + the run metadata, which is what's worth keeping next to the run.
    payload = {
        "checkpoint": str(ckpt_path),
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "results": results.get("results", {}),
        "n_params": n_params,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
