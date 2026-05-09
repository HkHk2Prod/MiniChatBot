"""Shared training runner for pretrain, SFT, and (eventually) RL.

The thin scripts in `scripts/` only handle CLI parsing + checkpoint-arg
resolution; everything from "build tokenizer" through "trainer.fit()"
lives here so adding a new training stage is a new entry-point script,
not another copy of this scaffolding.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from minichatbot.config import Config, save_config
from minichatbot.data import DATASET_REGISTRY
from minichatbot.data.collators import COLLATOR_REGISTRY
from minichatbot.model import MODEL_REGISTRY
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer import TOKENIZER_REGISTRY
from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.losses import LOSS_REGISTRY
from minichatbot.training.optim import build_optimizer, build_scheduler
from minichatbot.training.trainer import Trainer
from minichatbot.utils.torch_helpers import resolve_device


def make_run_dir(cfg: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{ts}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_and_train(
    cfg: Config,
    *,
    dataset_key: str,
    collator_key: str,
    loss_key: str,
    pretrained_ckpt: Path | None = None,
    resume_ckpt: Path | None = None,
) -> None:
    """Build everything from `cfg` and run `Trainer.fit()`.

    `pretrained_ckpt` loads ONLY model weights (SFT bootstrap); step
    counter and optimizer state stay fresh. `resume_ckpt` restores full
    training state. The two are mutually exclusive — callers should
    enforce that before calling.
    """
    if pretrained_ckpt is not None and resume_ckpt is not None:
        raise ValueError(
            "build_and_train: pretrained_ckpt and resume_ckpt are mutually exclusive."
        )

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

    ds_cls = DATASET_REGISTRY[dataset_key]
    train_ds = ds_cls.from_config(cfg.data, tokenizer, split="train")
    val_ds = (
        ds_cls.from_config(cfg.data, tokenizer, split="val")
        if cfg.data.val_path
        else None
    )

    coll_cls = COLLATOR_REGISTRY[collator_key]
    collator = coll_cls.from_config(tokenizer)

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
    if pretrained_ckpt is not None:
        print(f"loading pretrain weights from {pretrained_ckpt}")
        state = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])

    if cfg.trainer.compile:
        # torch.compile returns an OptimizedModule wrapper. At runtime it
        # delegates attribute access (cfg, parameters, etc.) to the wrapped
        # model, so it's still effectively a LanguageModel — but PyTorch's
        # stubs don't preserve the type, so cast to keep the rest of the
        # pipeline (Trainer, build_optimizer) statically typed.
        model = cast(LanguageModel, torch.compile(model))

    loss_cls = LOSS_REGISTRY[loss_key]
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

    if resume_ckpt is not None:
        print(f"resuming from {resume_ckpt} (will continue past step {trainer.step})")
        trainer.load_checkpoint(resume_ckpt, map_location=device)
        print(f"resumed at step {trainer.step}")

    trainer.fit()
