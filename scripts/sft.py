"""Supervised fine-tuning (SFT) on a chat-format JSONL.

Usage:
    # Bootstrap from the latest pretrain run with a ckpt_best.pt
    python scripts/sft.py --config configs/sft.yaml --from-pretrained auto

    # Or point at a specific run / checkpoint
    python scripts/sft.py --config configs/sft.yaml --from-pretrained runs/.../ckpt_best.pt
    python scripts/sft.py --config configs/sft.yaml --from-pretrained runs/<dir>

    # Filter --from-pretrained=auto to a specific pretrain run name
    python scripts/sft.py --config configs/sft.yaml --from-pretrained auto \\
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
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from minichatbot.config import Config, load_config, save_config
from minichatbot.data import DATASET_REGISTRY
from minichatbot.data.collators import COLLATOR_REGISTRY
from minichatbot.model import MODEL_REGISTRY
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer import TOKENIZER_REGISTRY
from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.losses import LOSS_REGISTRY
from minichatbot.training.optim import build_optimizer, build_scheduler
from minichatbot.training.trainer import Trainer
from minichatbot.utils.checkpoints import (
    find_best_checkpoint,
    find_best_checkpoint_in,
    find_latest_checkpoint,
    find_latest_checkpoint_in,
)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def make_run_dir(cfg: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{ts}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_from_pretrained(arg: str, cfg: Config, run_name_filter: str | None) -> Path:
    """Turn --from-pretrained into a concrete .pt path, preferring `ckpt_best.pt`."""
    if arg == "auto":
        ckpt = find_best_checkpoint_in(cfg.output_dir, run_name=run_name_filter)
        if ckpt is None:
            ckpt = find_latest_checkpoint_in(cfg.output_dir, run_name=run_name_filter)
        if ckpt is None:
            raise SystemExit(
                f"--from-pretrained auto: no checkpoints found under {cfg.output_dir}"
                + (f" for run_name='{run_name_filter}'" if run_name_filter else "")
                + ". Run pretrain first, or pass an explicit path."
            )
        return ckpt
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        ckpt = find_best_checkpoint(p) or find_latest_checkpoint(p)
        if ckpt is None:
            raise SystemExit(f"--from-pretrained: no checkpoints in {p}/checkpoints/")
        return ckpt
    raise SystemExit(f"--from-pretrained path does not exist: {p}")


def resolve_resume(arg: str, cfg: Config) -> Path:
    """Turn --resume into a concrete .pt path. Same shape as pretrain.py."""
    if arg == "auto":
        ckpt = find_latest_checkpoint_in(cfg.output_dir, run_name=cfg.run_name)
        if ckpt is None:
            raise SystemExit(
                f"--resume auto: no checkpoints found under {cfg.output_dir} "
                f"matching run_name='{cfg.run_name}'."
            )
        return ckpt
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        ckpt = find_latest_checkpoint(p)
        if ckpt is None:
            raise SystemExit(f"--resume: no checkpoints in {p}/checkpoints/")
        return ckpt
    raise SystemExit(f"--resume path does not exist: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--loss", default="sft", help="LOSS_REGISTRY key.")
    parser.add_argument("--collator", default="sft", help="COLLATOR_REGISTRY key.")
    parser.add_argument("--dataset", default="sft", help="DATASET_REGISTRY key.")
    parser.add_argument(
        "--from-pretrained",
        default=None,
        help=(
            "Bootstrap from a pretrained checkpoint .pt OR run dir. "
            "Use 'auto' to pick the latest run with a ckpt_best.pt. "
            "Loads ONLY model weights — optimizer/scheduler/step start fresh."
        ),
    )
    parser.add_argument(
        "--pretrain-run-name",
        default=None,
        help="When --from-pretrained=auto, only consider runs ending with _{name}.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume an interrupted SFT run (full state: weights + optimizer + step). "
            "Mutually exclusive with --from-pretrained."
        ),
    )
    args = parser.parse_args()

    if args.from_pretrained and args.resume:
        raise SystemExit(
            "--from-pretrained and --resume are mutually exclusive. "
            "from-pretrained starts a new SFT trajectory; resume continues an existing one."
        )

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")

    tok_cls = TOKENIZER_REGISTRY[cfg.tokenizer.type]
    tokenizer = tok_cls.from_config(cfg.tokenizer)
    tokenizer.save(run_dir / "tokenizer.json")

    ds_cls = DATASET_REGISTRY[args.dataset]
    train_ds = ds_cls.from_config(cfg.data, tokenizer, split="train")
    val_ds = (
        ds_cls.from_config(cfg.data, tokenizer, split="val")
        if cfg.data.val_path
        else None
    )

    coll_cls = COLLATOR_REGISTRY[args.collator]
    # The SFT collator needs pad_id; the pretrain collator doesn't take args.
    # Branch by key for now — when a third collator lands, refactor to a
    # tokenizer-aware Collator.from_config interface.
    collator = (
        coll_cls(pad_id=tokenizer.pad_id) if args.collator == "sft" else coll_cls()
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.data.num_workers,
            pin_memory=pin,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    model_cls = MODEL_REGISTRY[cfg.model.type]
    model = model_cls.from_config(cfg.model).to(device)

    # Bootstrap from pretrain BEFORE optimizer/scheduler are built so the
    # optimizer's parameter list is correct for the (possibly torch.compiled)
    # model. Step counter intentionally stays at 0.
    if args.from_pretrained is not None:
        ckpt = resolve_from_pretrained(args.from_pretrained, cfg, args.pretrain_run_name)
        print(f"loading pretrain weights from {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])

    if cfg.trainer.compile:
        model = cast(LanguageModel, torch.compile(model))

    loss_cls = LOSS_REGISTRY[args.loss]
    loss_fn = loss_cls().to(device)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.trainer.max_steps)

    callbacks = []
    for spec in cfg.callbacks:
        cls = CALLBACK_REGISTRY[spec.type]
        callbacks.append(cls(**spec.params))

    trainer = Trainer(
        config=cfg.trainer,
        full_config=cfg,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
        run_dir=run_dir,
        device=device,
        tokenizer=tokenizer,
    )

    if args.resume is not None:
        ckpt = resolve_resume(args.resume, cfg)
        print(f"resuming SFT from {ckpt} (will continue past step {trainer.step})")
        trainer.load_checkpoint(ckpt, map_location=device)
        print(f"resumed at step {trainer.step}")

    trainer.fit()


if __name__ == "__main__":
    main()
