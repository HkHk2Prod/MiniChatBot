"""Shared, config-driven training runner.

`scripts/train/train.py` only parses CLI args + resolves checkpoint paths;
everything from "build tokenizer" through "trainer.fit()" lives here (with
the per-step pieces in `builders.py`). The stage comes from `cfg.stage`,
which selects the trainer via `TRAINER_BUILDERS` and the default dataset/
collator/loss keys via `STAGE_DEFAULTS` — so adding a training stage is a
new row in each plus (if it needs a custom step) a `Trainer` subclass.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from minichatbot.config import Config, save_config
from minichatbot.inference.generator import Generator
from minichatbot.rl.rewards import REWARD_REGISTRY
from minichatbot.tokenizer import Tokenizer
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
from minichatbot.training.dpo_trainer import DPOTrainer
from minichatbot.training.losses.dpo import DPOLoss
from minichatbot.training.optim import build_optimizer, build_scheduler
from minichatbot.training.rl_trainer import GRPOTrainer
from minichatbot.training.trainer import Trainer
from minichatbot.utils.torch_helpers import resolve_device, unwrap_compiled

# Per-stage default registry keys. `cfg.stage` selects the row; the optional
# `cfg.dataset`/`cfg.collator`/`cfg.loss` fields override individual entries.
# Extend the pipeline by adding a row here + a TRAINER_BUILDERS entry.
STAGE_DEFAULTS: dict[str, dict[str, str]] = {
    "pretrain": {"dataset": "pretrain", "collator": "pretrain", "loss": "pretrain"},
    # DAPT (domain-adaptive pretraining) is continued next-token prediction on
    # a tailored corpus — same machinery as pretrain, distinct stage name so a
    # branch reads pretrain -> dapt -> dpo and each run is self-labelling.
    "dapt": {"dataset": "pretrain", "collator": "pretrain", "loss": "pretrain"},
    "sft": {"dataset": "sft", "collator": "sft", "loss": "sft"},
    "rl": {"dataset": "rl", "collator": "rl", "loss": "grpo"},
    "dpo": {"dataset": "mc", "collator": "mc", "loss": "dpo"},
}

# Stages with no in-trainer validation pass (no val loader; held-out eval
# happens out-of-band). RL has no val; DPO's loss needs reference log-probs
# the generic eval callback can't supply, so it's evaluated via lm-eval too.
STAGES_WITHOUT_VAL: set[str] = {"rl", "dpo"}


def _build_supervised_trainer(
    cfg: Config,
    *,
    common_kwargs: dict[str, Any],
    val_loader: Any,
    tokenizer: Tokenizer,
) -> Trainer:
    return Trainer(val_loader=val_loader, **common_kwargs)


def _build_grpo_trainer(
    cfg: Config,
    *,
    common_kwargs: dict[str, Any],
    val_loader: Any,
    tokenizer: Tokenizer,
) -> Trainer:
    im_end_id = tokenizer.special_token_id(IM_END_TOKEN)
    if im_end_id is None:
        raise ValueError(
            "RL needs the chat <|im_end|> token to know when a sampled "
            "completion has ended. Train/load a tokenizer that includes it "
            "(default in BPETokenizer.DEFAULT_SPECIALS)."
        )
    # `Generator.eos_id` is "whatever EOS the caller hands me"; here it's the
    # chat turn-end so completions terminate cleanly. On the trainer side the
    # same id is `chat_end_id` because that's what it represents in RL.
    generator = Generator(strategy=build_sampling_strategy(cfg.rl), eos_id=im_end_id)
    reward_fn = REWARD_REGISTRY[cfg.rl.reward]()
    return GRPOTrainer(
        rl_config=cfg.rl,
        generator=generator,
        reward_fn=reward_fn,
        chat_end_id=im_end_id,
        val_loader=None,
        **common_kwargs,
    )


def _build_dpo_trainer(
    cfg: Config,
    *,
    common_kwargs: dict[str, Any],
    val_loader: Any,
    tokenizer: Tokenizer,
) -> Trainer:
    # The DPO reference is a frozen snapshot of the initial (from-pretrained)
    # policy — clone its current weights before any optimizer step. Cloning
    # the uncompiled inner module keeps the copy plain (no torch.compile
    # wrapper); freezing keeps it out of grad/optimizer.
    policy = common_kwargs["model"]
    device = common_kwargs["device"]
    ref_model = copy.deepcopy(unwrap_compiled(policy)).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    # The generic build_loss made a default DPOLoss; swap in the configured one.
    kwargs = {**common_kwargs, "loss": DPOLoss(beta=cfg.dpo.beta, score_norm=cfg.dpo.score_norm)}
    return DPOTrainer(ref_model=ref_model, val_loader=val_loader, **kwargs)


# stage -> a callable that constructs its trainer from the common kwargs.
TRAINER_BUILDERS: dict[str, Callable[..., Trainer]] = {
    "pretrain": _build_supervised_trainer,
    "dapt": _build_supervised_trainer,
    "sft": _build_supervised_trainer,
    "rl": _build_grpo_trainer,
    "dpo": _build_dpo_trainer,
}


def build_and_train(
    cfg: Config,
    *,
    pretrained_ckpt: Path | None = None,
    resume_ckpt: Path | None = None,
) -> None:
    """Build everything from `cfg` and run `Trainer.fit()`.

    The stage is `cfg.stage`: it selects the trainer (`TRAINER_BUILDERS`)
    and the default (dataset, collator, loss) registry keys
    (`STAGE_DEFAULTS`), which `cfg.dataset`/`cfg.collator`/`cfg.loss` may
    individually override.

    `pretrained_ckpt` loads ONLY model weights (a fresh trajectory on top
    of a previous stage's checkpoint); `resume_ckpt` restores full training
    state. Mutually exclusive.
    """
    stage = cfg.stage or ""
    if stage not in STAGE_DEFAULTS or stage not in TRAINER_BUILDERS:
        raise ValueError(
            f"Unknown training stage {stage!r}; known stages: "
            f"{sorted(STAGE_DEFAULTS)}. Set `stage:` in the config."
        )
    defaults = STAGE_DEFAULTS[stage]
    dataset_key = cfg.dataset or defaults["dataset"]
    collator_key = cfg.collator or defaults["collator"]
    loss_key = cfg.loss or defaults["loss"]

    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")

    tokenizer = build_tokenizer(cfg, run_dir)
    train_loader, val_loader = build_loaders(
        cfg, tokenizer, dataset_key=dataset_key, collator_key=collator_key,
        device=device, with_val=(stage not in STAGES_WITHOUT_VAL),
    )

    incoming_state, effective_model_cfg, startup_warnings = prepare_model_state(
        cfg, resume_ckpt=resume_ckpt, pretrained_ckpt=pretrained_ckpt, device=device,
    )
    model = build_model(
        effective_model_cfg, device=device, compile=cfg.trainer.compile,
        pretrained_ckpt=pretrained_ckpt, incoming_state=incoming_state,
        weights_label="previous-stage",
        grad_checkpointing=cfg.trainer.grad_checkpointing,
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

    trainer = TRAINER_BUILDERS[stage](
        cfg, common_kwargs=common_kwargs, val_loader=val_loader, tokenizer=tokenizer
    )

    if resume_ckpt is not None:
        print(
            f"resuming {stage} from {resume_ckpt} "
            f"(will continue past step {trainer.step})"
        )
        trainer.load_checkpoint(
            resume_ckpt, map_location=device, preloaded_state=incoming_state
        )
        print(f"resumed at step {trainer.step}")

    trainer.fit()
