"""Pretrain a small language model from a YAML config.

Usage:
    python scripts/train/pretrain.py --config configs/pretrain_small.yaml

Builds tokenizer, datasets, model, optimizer, scheduler, loss, and
callbacks from registries based on the YAML; runs Trainer.fit().

Resume an interrupted run with --resume (full state: weights + optimizer + step):
    python scripts/train/pretrain.py --config configs/foo.yaml --resume runs/.../ckpt.pt
    python scripts/train/pretrain.py --config configs/foo.yaml --resume auto

Continue past a finished run with --from-pretrained (weights only; fresh
optimizer/scheduler/step — useful when --resume can't extend because the
old run already reached max_steps):
    python scripts/train/pretrain.py --config configs/foo.yaml --from-pretrained auto \\
        --pretrain-run-name pretrain_tinystories

A new run dir is created either way; the original is left untouched.
"""

from __future__ import annotations

import argparse

from minichatbot.config import load_config
from minichatbot.training.cli import add_train_args, resolve_train_ckpts
from minichatbot.training.runner import build_and_train


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
