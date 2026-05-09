"""Pretrain a small language model from a YAML config.

Usage:
    python scripts/train/pretrain.py --config configs/pretrain_small.yaml

Builds tokenizer, datasets, model, optimizer, scheduler, loss, and
callbacks from registries based on the YAML; runs Trainer.fit().

Resume from a previous run with --resume:
    python scripts/train/pretrain.py --config configs/foo.yaml --resume runs/.../ckpt.pt
    python scripts/train/pretrain.py --config configs/foo.yaml --resume auto
A new run dir is created for the resumed run; the original is left untouched.
"""

from __future__ import annotations

import argparse

from minichatbot.config import load_config
from minichatbot.training.cli import add_train_args
from minichatbot.training.runner import build_and_train
from minichatbot.utils.checkpoints import resolve_resume_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small language model.")
    add_train_args(
        parser,
        default_loss="pretrain",
        default_collator="pretrain",
        default_dataset="pretrain",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    resume_ckpt = resolve_resume_arg(args.resume, cfg) if args.resume else None

    build_and_train(
        cfg,
        dataset_key=args.dataset,
        collator_key=args.collator,
        loss_key=args.loss,
        resume_ckpt=resume_ckpt,
    )


if __name__ == "__main__":
    main()
