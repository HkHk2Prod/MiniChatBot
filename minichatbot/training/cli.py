"""Shared CLI helpers for training scripts (pretrain, sft, future RL)."""

from __future__ import annotations

import argparse


def add_train_args(
    parser: argparse.ArgumentParser,
    *,
    default_loss: str,
    default_collator: str,
    default_dataset: str,
) -> None:
    """Flags shared by all YAML-config trainers: config path, registry keys, resume.

    The three registry keys (loss/collator/dataset) default to per-script values
    (e.g. "pretrain" or "sft"); override only when mixing components across stages.
    """
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--loss", default=default_loss, help="LOSS_REGISTRY key.")
    parser.add_argument("--collator", default=default_collator, help="COLLATOR_REGISTRY key.")
    parser.add_argument("--dataset", default=default_dataset, help="DATASET_REGISTRY key.")
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume from a checkpoint .pt OR a run dir (latest ckpt inside). "
            "Use 'auto' to pick the latest run matching cfg.run_name. "
            "A new run dir is created for the resumed run; original stays put."
        ),
    )
