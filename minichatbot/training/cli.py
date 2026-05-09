"""Shared CLI helpers for training scripts (pretrain, sft, future RL)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from minichatbot.utils.checkpoints import resolve_pretrained_arg, resolve_resume_arg

if TYPE_CHECKING:
    from minichatbot.config import Config


def add_train_args(
    parser: argparse.ArgumentParser,
    *,
    default_loss: str,
    default_collator: str,
    default_dataset: str,
) -> None:
    """Flags shared by all YAML-config trainers: config, registry keys,
    --resume, --from-pretrained, --pretrain-run-name.

    Registry keys default to per-script values (e.g. "pretrain" or "sft");
    override only when mixing components across stages.
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
    parser.add_argument(
        "--from-pretrained",
        default=None,
        help=(
            "Bootstrap from a checkpoint .pt OR run dir. Use 'auto' to pick "
            "the latest run with a ckpt_best.pt. Loads ONLY model weights — "
            "optimizer/scheduler/step start fresh. Mutually exclusive with --resume."
        ),
    )
    parser.add_argument(
        "--pretrain-run-name",
        default=None,
        help="When --from-pretrained=auto, only consider runs ending with _{name}.",
    )


def resolve_train_ckpts(
    args: argparse.Namespace,
    cfg: Config,
) -> tuple[Path | None, Path | None]:
    """Resolve --from-pretrained / --resume to concrete checkpoint paths.

    Returns ``(pretrained_ckpt, resume_ckpt)``. Raises ``SystemExit`` if both
    are set — they're mutually exclusive (from-pretrained starts a new
    trajectory, resume continues an existing one).
    """
    if args.from_pretrained and args.resume:
        raise SystemExit(
            "--from-pretrained and --resume are mutually exclusive. "
            "from-pretrained starts a new training trajectory; resume continues an existing one."
        )
    pretrained_ckpt = (
        resolve_pretrained_arg(args.from_pretrained, cfg, args.pretrain_run_name)
        if args.from_pretrained
        else None
    )
    resume_ckpt = resolve_resume_arg(args.resume, cfg) if args.resume else None
    return pretrained_ckpt, resume_ckpt
