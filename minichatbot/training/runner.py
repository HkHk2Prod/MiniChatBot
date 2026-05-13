"""Shared training runner for pretrain and SFT.

The thin scripts in `scripts/` only handle CLI parsing + checkpoint-arg
resolution; everything from "build tokenizer" through "trainer.fit()"
lives here (and in `builders.py` for the pieces also shared with the RL
runner) so adding a training stage is a new entry-point script, not
another copy of this scaffolding.
"""

from __future__ import annotations

from pathlib import Path

import torch

from minichatbot.config import Config, save_config
from minichatbot.training.builders import (
    build_callbacks,
    build_loaders,
    build_loss,
    build_model,
    build_tokenizer,
    make_run_dir,
    preload_checkpoint,
)
from minichatbot.training.optim import build_optimizer, build_scheduler
from minichatbot.training.trainer import Trainer
from minichatbot.utils.torch_helpers import resolve_device

__all__ = ["build_and_train", "make_run_dir"]


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
    training state. The two are mutually exclusive.
    """
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")

    tokenizer = build_tokenizer(cfg, run_dir)
    train_loader, val_loader = build_loaders(
        cfg, tokenizer, dataset_key=dataset_key, collator_key=collator_key,
        device=device, with_val=True,
    )

    incoming_state, effective_model_cfg, startup_warnings = preload_checkpoint(
        cfg, resume_ckpt=resume_ckpt, pretrained_ckpt=pretrained_ckpt, device=device,
    )
    model = build_model(
        effective_model_cfg, device=device, compile=cfg.trainer.compile,
        pretrained_ckpt=pretrained_ckpt, incoming_state=incoming_state,
        weights_label="pretrain",
    )

    loss_fn = build_loss(loss_key, device)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.trainer.max_steps)

    trainer = Trainer(
        config=cfg.trainer,
        full_config=cfg,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=build_callbacks(cfg),
        run_dir=run_dir,
        device=device,
        tokenizer=tokenizer,
        startup_warnings=startup_warnings,
    )

    if resume_ckpt is not None:
        print(f"resuming from {resume_ckpt} (will continue past step {trainer.step})")
        trainer.load_checkpoint(
            resume_ckpt, map_location=device, preloaded_state=incoming_state
        )
        print(f"resumed at step {trainer.step}")

    trainer.fit()
