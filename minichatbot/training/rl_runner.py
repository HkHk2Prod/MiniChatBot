"""Runner for the RL (GRPO) stage — the `build_and_train` of `scripts/train/rl.py`.

Parallels `minichatbot.training.runner.build_and_train` but wires the
RL-specific pieces: a `Generator` (for sampling rollouts), a `Reward`,
the `rl` prompt dataset + collator, and `GRPOTrainer` instead of the
plain `Trainer`. Everything that isn't RL-specific comes from
`minichatbot.training.builders`. The thin `scripts/train/rl.py` only does
CLI parsing and checkpoint-arg resolution.

RL almost always starts from an SFT checkpoint via `--from-pretrained`
(loads model weights only; optimizer/scheduler/step start fresh —
exactly the SFT semantics, one stage further along). `--resume`
continues a paused RL run.
"""

from __future__ import annotations

from pathlib import Path

import torch

from minichatbot.config import Config, RLConfig, save_config
from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies.base import SamplingStrategy
from minichatbot.inference.strategies.temperature import TemperatureSampling
from minichatbot.inference.strategies.top_k import TopKSampling
from minichatbot.inference.strategies.top_p import TopPSampling
from minichatbot.rl.rewards.base import Reward
from minichatbot.tokenizer.bpe import IM_END_TOKEN
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
from minichatbot.training.rl_trainer import GRPOTrainer
from minichatbot.utils.torch_helpers import resolve_device


def _build_sampling_strategy(rl: RLConfig) -> SamplingStrategy:
    """Rollout sampler: top-k if set, else nucleus if top_p < 1, else plain
    temperature sampling. Greedy (temperature=0) would make every completion
    in a group identical → zero advantage everywhere, so don't do that."""
    if rl.top_k is not None:
        return TopKSampling(k=rl.top_k, temperature=rl.temperature)
    if rl.top_p < 1.0:
        return TopPSampling(p=rl.top_p, temperature=rl.temperature)
    return TemperatureSampling(temperature=rl.temperature)


def build_and_train_rl(
    cfg: Config,
    *,
    dataset_key: str,
    collator_key: str,
    loss_key: str,
    pretrained_ckpt: Path | None = None,
    resume_ckpt: Path | None = None,
) -> None:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")

    tokenizer = build_tokenizer(cfg, run_dir)

    im_end_id = tokenizer.special_token_id(IM_END_TOKEN)
    if im_end_id is None:
        raise ValueError(
            "RL needs the chat <|im_end|> token to know when a sampled "
            "completion has ended. Train/load a tokenizer that includes it "
            "(default in BPETokenizer.DEFAULT_SPECIALS)."
        )

    train_loader, _ = build_loaders(
        cfg, tokenizer, dataset_key=dataset_key, collator_key=collator_key,
        device=device, with_val=False,
    )

    incoming_state, effective_model_cfg, startup_warnings = preload_checkpoint(
        cfg, resume_ckpt=resume_ckpt, pretrained_ckpt=pretrained_ckpt, device=device,
    )
    model = build_model(
        effective_model_cfg, device=device, compile=cfg.trainer.compile,
        pretrained_ckpt=pretrained_ckpt, incoming_state=incoming_state,
        weights_label="SFT",
    )

    loss_fn = build_loss(loss_key, device)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.trainer.max_steps)

    reward_fn: Reward = Reward.from_key(cfg.rl.reward)
    generator = Generator(strategy=_build_sampling_strategy(cfg.rl), eos_id=im_end_id)

    trainer = GRPOTrainer(
        rl_config=cfg.rl,
        generator=generator,
        reward_fn=reward_fn,
        eos_id=im_end_id,
        config=cfg.trainer,
        full_config=cfg,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=None,
        callbacks=build_callbacks(cfg),
        run_dir=run_dir,
        device=device,
        tokenizer=tokenizer,
        startup_warnings=startup_warnings,
    )

    if resume_ckpt is not None:
        print(f"resuming RL from {resume_ckpt} (will continue past step {trainer.step})")
        trainer.load_checkpoint(resume_ckpt, map_location=device, preloaded_state=incoming_state)
        print(f"resumed at step {trainer.step}")

    trainer.fit()
