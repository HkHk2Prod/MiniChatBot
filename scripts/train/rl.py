"""Reinforcement learning (GRPO) on a prompt set with a verifiable reward.

Usage:
    # Start from the latest SFT run with a ckpt_best.pt
    python scripts/train/rl.py --config configs/rl_gsm8k.yaml --from-pretrained auto

    # Or point at a specific SFT run / checkpoint
    python scripts/train/rl.py --config configs/rl_gsm8k.yaml \\
        --from-pretrained runs/<sft_dir>/ckpt_best.pt

    # Filter --from-pretrained=auto to a specific SFT run name
    python scripts/train/rl.py --config configs/rl_gsm8k.yaml --from-pretrained auto \\
        --pretrain-run-name sft_tinystories

    # Continue a paused RL run
    python scripts/train/rl.py --config configs/rl_gsm8k.yaml --resume auto

Defaults: --dataset rl, --collator rl, --loss grpo. The RL knobs
(group_size, max_new_tokens, sampling temperature/top-p, reward name)
live under the `rl:` section of the YAML config.

`--from-pretrained` loads only model weights — optimizer state, scheduler
state, and step counter all start fresh; that's the RL semantic (a new
training trajectory on top of the SFT policy).

Expected prompt JSONL (one task per line):
    {"question": "Natalia sold ...", "answer": "She sold ... #### 72"}
The reward function (cfg.rl.reward) decides how a sampled completion is
scored against `answer` — see `minichatbot.rl.REWARD_REGISTRY`. Build a
GSM8K-formatted set with `scripts/data/download_rl_data.py`.
"""

from __future__ import annotations

import argparse

from minichatbot.config import load_config
from minichatbot.training.cli import add_train_args, resolve_train_ckpts
from minichatbot.training.rl_runner import build_and_train_rl


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO reinforcement learning.")
    add_train_args(
        parser,
        default_loss="grpo",
        default_collator="rl",
        default_dataset="rl",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pretrained_ckpt, resume_ckpt = resolve_train_ckpts(args, cfg)

    build_and_train_rl(
        cfg,
        dataset_key=args.dataset,
        collator_key=args.collator,
        loss_key=args.loss,
        pretrained_ckpt=pretrained_ckpt,
        resume_ckpt=resume_ckpt,
    )


if __name__ == "__main__":
    main()
