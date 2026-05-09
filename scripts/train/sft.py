"""Supervised fine-tuning (SFT) on a chat-format JSONL.

Usage:
    # Bootstrap from the latest pretrain run with a ckpt_best.pt
    python scripts/train/sft.py --config configs/sft.yaml --from-pretrained auto

    # Or point at a specific run / checkpoint
    python scripts/train/sft.py --config configs/sft.yaml --from-pretrained runs/.../ckpt_best.pt
    python scripts/train/sft.py --config configs/sft.yaml --from-pretrained runs/<dir>

    # Filter --from-pretrained=auto to a specific pretrain run name
    python scripts/train/sft.py --config configs/sft.yaml --from-pretrained auto \\
        --pretrain-run-name pretrain_tinystories

Defaults: --dataset sft, --collator sft, --loss sft. Override only when
you know what you're doing.

`--from-pretrained` loads only model weights — optimizer state, scheduler
state, and step counter all start fresh. That's the SFT semantic: a new
training trajectory built on top of pretrained representations. Use
`--resume` (same shape as pretrain.py) to continue a paused SFT run.

Expected JSONL format (one line per conversation):
    {"messages": [
        {"role": "system",    "content": "You are helpful."},
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."}
    ]}
"""

from __future__ import annotations

import argparse

from minichatbot.config import load_config
from minichatbot.training.cli import add_train_args
from minichatbot.training.runner import build_and_train
from minichatbot.utils.checkpoints import resolve_pretrained_arg, resolve_resume_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning.")
    add_train_args(
        parser,
        default_loss="sft",
        default_collator="sft",
        default_dataset="sft",
    )
    parser.add_argument(
        "--from-pretrained",
        default=None,
        help=(
            "Bootstrap from a pretrained checkpoint .pt OR run dir. "
            "Use 'auto' to pick the latest run with a ckpt_best.pt. "
            "Loads ONLY model weights — optimizer/scheduler/step start fresh. "
            "Mutually exclusive with --resume."
        ),
    )
    parser.add_argument(
        "--pretrain-run-name",
        default=None,
        help="When --from-pretrained=auto, only consider runs ending with _{name}.",
    )
    args = parser.parse_args()

    if args.from_pretrained and args.resume:
        raise SystemExit(
            "--from-pretrained and --resume are mutually exclusive. "
            "from-pretrained starts a new SFT trajectory; resume continues an existing one."
        )

    cfg = load_config(args.config)
    pretrained_ckpt = (
        resolve_pretrained_arg(args.from_pretrained, cfg, args.pretrain_run_name)
        if args.from_pretrained
        else None
    )
    resume_ckpt = resolve_resume_arg(args.resume, cfg) if args.resume else None

    build_and_train(
        cfg,
        dataset_key=args.dataset,
        collator_key=args.collator,
        loss_key=args.loss,
        pretrained_ckpt=pretrained_ckpt,
        resume_ckpt=resume_ckpt,
    )


if __name__ == "__main__":
    main()
