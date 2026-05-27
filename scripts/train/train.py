"""Unified training launcher — runs any stage from a single YAML config.

    python scripts/train/train.py --config configs/100M/pretrain_fineweb.yaml
    python scripts/train/train.py --config configs/100M/sft_fineweb.yaml \\
        --from-pretrained auto --pretrain-run-name pretrain_fineweb
    python scripts/train/train.py --config configs/100M/rl_gsm8k.yaml --resume auto

The stage (pretrain | sft | rl | ...) comes from the config's `stage:` field
(falling back to `data.type`). It selects the trainer and the default
dataset/collator/loss registry keys; `--stage` / `--dataset` / `--collator` /
`--loss` override individual choices. `--from-pretrained` loads weights only
(fresh optimizer/scheduler/step — a new trajectory on a previous stage);
`--resume` restores full training state. A new run dir is created either way.
"""

from __future__ import annotations

from minichatbot.training.cli import train_main

if __name__ == "__main__":
    train_main()
