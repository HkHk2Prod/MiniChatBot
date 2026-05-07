"""Pretrain a small language model from a YAML config.

Usage:
    python scripts/pretrain.py --config configs/pretrain_small.yaml

Builds tokenizer, datasets, model, optimizer, scheduler, loss, and
callbacks from registries based on the YAML; runs Trainer.fit().

Resume from a previous run with --resume:
    python scripts/pretrain.py --config configs/foo.yaml --resume runs/.../ckpt.pt
    python scripts/pretrain.py --config configs/foo.yaml --resume auto
A new run dir is created for the resumed run; the original is left untouched.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small language model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--loss", default="pretrain", help="LOSS_REGISTRY key.")
    parser.add_argument("--collator", default="pretrain", help="COLLATOR_REGISTRY key.")
    parser.add_argument("--dataset", default="pretrain", help="DATASET_REGISTRY key.")
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume from a checkpoint .pt OR a run dir (latest ckpt inside). "
            "Use 'auto' to pick the latest run matching cfg.run_name. "
            "A new run dir is created for the resumed run; original stays put."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")

    tok_cls = TOKENIZER_REGISTRY[cfg.tokenizer.type]
    tokenizer = tok_cls.from_config(cfg.tokenizer)

    # Snapshot the tokenizer in the run dir so the run is self-contained:
    # generate.py / future SFT loaders can find it next to checkpoints
    # without needing the original data/ tree to still exist.
    tokenizer.save(run_dir / "tokenizer.json")

    ds_cls = DATASET_REGISTRY[args.dataset]
    train_ds = ds_cls.from_config(cfg.data, tokenizer, split="train")
    val_ds = (
        ds_cls.from_config(cfg.data, tokenizer, split="val")
        if cfg.data.val_path
        else None
    )

    coll_cls = COLLATOR_REGISTRY[args.collator]
    collator = coll_cls()

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
    if cfg.trainer.compile:
        # torch.compile returns an OptimizedModule wrapper. At runtime it
        # delegates attribute access (cfg, parameters, etc.) to the wrapped
        # model, so it's still effectively a LanguageModel — but PyTorch's
        # stubs don't preserve the type, so cast to keep the rest of the
        # pipeline (Trainer, build_optimizer) statically typed.
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
        ckpt = _resolve_resume(args.resume, cfg)
        print(f"resuming from {ckpt} (will continue past step {trainer.step})")
        trainer.load_checkpoint(ckpt, map_location=device)
        print(f"resumed at step {trainer.step}")

    trainer.fit()


def _resolve_resume(arg: str, cfg: Config) -> Path:
    """Turn a --resume CLI value into a concrete checkpoint path."""
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


if __name__ == "__main__":
    main()
