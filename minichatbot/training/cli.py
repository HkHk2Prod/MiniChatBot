"""Shared CLI helpers + the unified training entrypoint (`train_main`)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from minichatbot.utils.checkpoints import resolve_pretrained_arg, resolve_resume_arg

if TYPE_CHECKING:
    from minichatbot.config import Config


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Flags for the unified launcher: config, optional stage + registry-key
    overrides, --resume, --from-pretrained, --pretrain-run-name.

    `stage` defaults to `cfg.stage` (itself defaulting to `data.type`); the
    dataset/collator/loss keys default to the runner's per-stage table. The
    flags only override — needed when mixing components across stages.
    """
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--stage",
        default=None,
        help="Override cfg.stage (pretrain | sft | rl | ...). Default: from config.",
    )
    parser.add_argument("--loss", default=None, help="Override LOSS_REGISTRY key.")
    parser.add_argument("--collator", default=None, help="Override COLLATOR_REGISTRY key.")
    parser.add_argument("--dataset", default=None, help="Override DATASET_REGISTRY key.")
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


def train_main(argv: list[str] | None = None) -> None:
    """Unified training entrypoint. Parses args, loads the config, applies any
    CLI overrides (stage / dataset / collator / loss), resolves the
    checkpoint args, and hands off to the config-driven runner.

    Imports are local so importing this module for arg parsing alone stays
    cheap (the runner pulls in torch + the whole builder stack).
    """
    from minichatbot.config import load_config
    from minichatbot.training.runner import build_and_train

    parser = argparse.ArgumentParser(
        description="Train MiniChatBot — the stage comes from the config."
    )
    add_train_args(parser)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # CLI overrides win over the config's declared stage/component keys.
    if args.stage:
        cfg.stage = args.stage
    if args.dataset:
        cfg.dataset = args.dataset
    if args.collator:
        cfg.collator = args.collator
    if args.loss:
        cfg.loss = args.loss

    pretrained_ckpt, resume_ckpt = resolve_train_ckpts(args, cfg)
    build_and_train(cfg, pretrained_ckpt=pretrained_ckpt, resume_ckpt=resume_ckpt)
