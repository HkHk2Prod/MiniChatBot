"""Shared training runner for pretrain / SFT / RL.

The thin scripts in `scripts/` only handle CLI parsing + checkpoint-arg
resolution; everything from "build tokenizer" through "trainer.fit()"
lives here (and in `builders.py` for the per-step pieces) so adding a
training stage is mostly just a new entry-point script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from minichatbot.config import Config, save_config
from minichatbot.inference.generator import Generator
from minichatbot.rl.rewards import REWARD_REGISTRY
from minichatbot.tokenizer.bpe import IM_END_TOKEN
from minichatbot.training.builders import (
    build_callbacks,
    build_loaders,
    build_loss,
    build_model,
    build_sampling_strategy,
    build_tokenizer,
    make_run_dir,
    prepare_model_state,
)
from minichatbot.training.optim import build_optimizer, build_scheduler
from minichatbot.training.rl_trainer import GRPOTrainer
from minichatbot.training.trainer import Trainer
from minichatbot.utils.torch_helpers import resolve_device

Stage = Literal["pretrain", "sft", "rl"]


def build_and_train(
    cfg: Config,
    *,
    stage: Stage,
    dataset_key: str,
    collator_key: str,
    loss_key: str,
    pretrained_ckpt: Path | None = None,
    resume_ckpt: Path | None = None,
) -> None:
    """Build everything from `cfg` and run `Trainer.fit()`.

    `pretrained_ckpt` loads ONLY model weights (SFT bootstrap, or RL on
    top of SFT); step counter and optimizer state stay fresh.
    `resume_ckpt` restores full training state. Mutually exclusive.

    `stage` selects the trainer: "pretrain"/"sft" use the supervised
    `Trainer`; "rl" uses `GRPOTrainer` and additionally builds the
    rollout `Generator` + reward function.
    """
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")

    tokenizer = build_tokenizer(cfg, run_dir)
    train_loader, val_loader = build_loaders(
        cfg, tokenizer, dataset_key=dataset_key, collator_key=collator_key,
        device=device, with_val=(stage != "rl"),
    )

    incoming_state, effective_model_cfg, startup_warnings = prepare_model_state(
        cfg, resume_ckpt=resume_ckpt, pretrained_ckpt=pretrained_ckpt, device=device,
    )
    # Just a label for the "loading X weights from ..." log line. SFT/RL
    # bootstrap from the previous stage's checkpoint.
    weights_label = "SFT" if stage == "rl" else "pretrain"
    model = build_model(
        effective_model_cfg, device=device, compile=cfg.trainer.compile,
        pretrained_ckpt=pretrained_ckpt, incoming_state=incoming_state,
        weights_label=weights_label,
    )

    loss_fn = build_loss(loss_key, device)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.trainer.max_steps)

    common_kwargs: dict[str, Any] = dict(
        config=cfg.trainer,
        full_config=cfg,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        callbacks=build_callbacks(cfg),
        run_dir=run_dir,
        device=device,
        tokenizer=tokenizer,
        startup_warnings=startup_warnings,
    )

    trainer: Trainer
    if stage == "rl":
        im_end_id = tokenizer.special_token_id(IM_END_TOKEN)
        if im_end_id is None:
            raise ValueError(
                "RL needs the chat <|im_end|> token to know when a sampled "
                "completion has ended. Train/load a tokenizer that includes it "
                "(default in BPETokenizer.DEFAULT_SPECIALS)."
            )
        # `Generator.eos_id` is "whatever EOS the caller hands me" — the
        # generator itself doesn't know about chat. Here we pass the chat
        # turn-end so completions terminate cleanly; on the trainer side
        # the same id is `chat_end_id` because that's what it actually
        # represents within the RL pipeline.
        generator = Generator(
            strategy=build_sampling_strategy(cfg.rl), eos_id=im_end_id
        )
        reward_fn = REWARD_REGISTRY[cfg.rl.reward]()
        trainer = GRPOTrainer(
            rl_config=cfg.rl,
            generator=generator,
            reward_fn=reward_fn,
            chat_end_id=im_end_id,
            val_loader=None,
            **common_kwargs,
        )
    else:
        trainer = Trainer(val_loader=val_loader, **common_kwargs)

    if resume_ckpt is not None:
        stage_label = "RL " if stage == "rl" else ""
        print(
            f"resuming {stage_label}from {resume_ckpt} "
            f"(will continue past step {trainer.step})"
        )
        trainer.load_checkpoint(
            resume_ckpt, map_location=device, preloaded_state=incoming_state
        )
        print(f"resumed at step {trainer.step}")

    trainer.fit()
