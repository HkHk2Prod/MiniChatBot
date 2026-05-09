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
`--resume` to continue a paused SFT run.

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
from minichatbot.training.cli import add_train_args, resolve_train_ckpts
from minichatbot.training.runner import build_and_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning.")
    add_train_args(
        parser,
        default_loss="sft",
        default_collator="sft",
        default_dataset="sft",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pretrained_ckpt, resume_ckpt = resolve_train_ckpts(args, cfg)

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
