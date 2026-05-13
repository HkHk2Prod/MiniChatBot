"""Builders shared by the training runners.

Both `runner.py` (pretrain/SFT) and `rl_runner.py` (GRPO) go through the
same sequence: turn `Config` + a few registry keys into a tokenizer,
data loaders, a model, a loss, callbacks — then hand them to a trainer.
That sequence lives here so each runner is just orchestration and only
imports the registries it genuinely needs (the RL runner: none of them).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, RandomSampler

from minichatbot.config import Config, ModelConfig
from minichatbot.data import DATASET_REGISTRY
from minichatbot.data.collators import COLLATOR_REGISTRY
from minichatbot.model import MODEL_REGISTRY, LanguageModel
from minichatbot.tokenizer import TOKENIZER_REGISTRY, Tokenizer
from minichatbot.training.callbacks import CALLBACK_REGISTRY, Callback
from minichatbot.training.losses import LOSS_REGISTRY, Loss
from minichatbot.utils.model_config_check import (
    parse_ckpt_model_config,
    reconcile_model_config,
)


def make_run_dir(cfg: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{ts}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_tokenizer(cfg: Config, run_dir: Path) -> Tokenizer:
    """Build the tokenizer and snapshot it into `run_dir` so the run is
    self-contained — generate.py / downstream stages find it next to the
    checkpoints without needing the original data/ tree to still exist."""
    tok_cls = TOKENIZER_REGISTRY[cfg.tokenizer.type]
    tokenizer = tok_cls.from_config(cfg.tokenizer)
    tokenizer.save(run_dir / "tokenizer.json")
    return tokenizer


def build_loaders(
    cfg: Config,
    tokenizer: Tokenizer,
    *,
    dataset_key: str,
    collator_key: str,
    device: torch.device,
    with_val: bool,
) -> tuple[DataLoader, DataLoader | None]:
    """Build the train loader and, if `with_val` and `cfg.data.val_path` is
    set, a val loader (else `None` — e.g. RL has no validation pass)."""
    ds_cls = DATASET_REGISTRY[dataset_key]
    collator = COLLATOR_REGISTRY[collator_key].from_config(tokenizer)
    pin = device.type == "cuda"

    train_ds = ds_cls.from_config(cfg.data, tokenizer, split="train")
    # DataLoader(shuffle=True) uses RandomSampler(replacement=False), whose
    # __iter__ does torch.randperm(N).tolist() — materializing a Python list
    # of N ints (~36 bytes each). On FineWeb-scale packed corpora N ≈ 1.3B,
    # which is ~46 GB and OOMs before step 1. Replacement-sampling generates
    # indices in O(1)-memory chunks. We only switch for huge N (pretrain),
    # where collision probability is negligible (samples-consumed << N);
    # SFT-size corpora keep the standard without-replacement shuffle.
    huge_dataset_threshold = 10_000_000
    if len(train_ds) > huge_dataset_threshold:
        train_sampler: RandomSampler | None = RandomSampler(
            train_ds, replacement=True, num_samples=len(train_ds)
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        drop_last=True,
    )

    val_loader: DataLoader | None = None
    if with_val and cfg.data.val_path:
        val_loader = DataLoader(
            ds_cls.from_config(cfg.data, tokenizer, split="val"),
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.data.num_workers,
            pin_memory=pin,
            drop_last=False,
        )
    return train_loader, val_loader


def preload_checkpoint(
    cfg: Config,
    *,
    resume_ckpt: Path | None,
    pretrained_ckpt: Path | None,
    device: torch.device,
) -> tuple[dict[str, Any] | None, ModelConfig, list[str]]:
    """Load a `--resume` / `--from-pretrained` checkpoint's contents early so
    its saved `model_config` can be reconciled with the YAML BEFORE the
    model is built — a shape mismatch caught here beats a cryptic size
    error inside `load_state_dict`. The returned state is reused for the
    actual weight load, so we still `torch.load` only once per run.

    Returns `(incoming_state, effective_model_cfg, startup_warnings)`.
    `resume_ckpt` and `pretrained_ckpt` are mutually exclusive.
    """
    if resume_ckpt is not None and pretrained_ckpt is not None:
        raise ValueError("resume_ckpt and pretrained_ckpt are mutually exclusive.")

    incoming_path = resume_ckpt if resume_ckpt is not None else pretrained_ckpt
    if incoming_path is None:
        return None, cfg.model, []

    incoming_state = torch.load(incoming_path, map_location=device, weights_only=False)
    if "model_config" not in incoming_state:
        return incoming_state, cfg.model, []

    ckpt_model_cfg = parse_ckpt_model_config(incoming_state["model_config"])
    mode = "resume" if resume_ckpt is not None else "from_pretrained"
    effective_model_cfg, startup_warnings = reconcile_model_config(
        cfg.model, ckpt_model_cfg, mode=mode
    )
    return incoming_state, effective_model_cfg, startup_warnings


def build_model(
    model_cfg: ModelConfig,
    *,
    device: torch.device,
    compile: bool,
    pretrained_ckpt: Path | None,
    incoming_state: dict[str, Any] | None,
    weights_label: str,
) -> LanguageModel:
    """Build the model and, if `pretrained_ckpt` is set, load ONLY its
    weights (step counter / optimizer stay fresh — the SFT/RL bootstrap).
    Done before the optimizer is built so the optimizer's parameter list
    matches the (possibly `torch.compile`d) model. `weights_label` only
    flavours the log line ('pretrain', 'SFT')."""
    model = MODEL_REGISTRY[model_cfg.type].from_config(model_cfg).to(device)

    if pretrained_ckpt is not None:
        print(f"loading {weights_label} weights from {pretrained_ckpt}")
        assert incoming_state is not None  # guaranteed by preload_checkpoint
        model.load_state_dict(incoming_state["model"])

    if compile:
        # torch.compile returns an OptimizedModule wrapper that delegates
        # attribute access (cfg, parameters, ...) to the wrapped model, so
        # it's still effectively a LanguageModel — but the stubs drop the
        # type, so cast to keep the downstream pipeline statically typed.
        model = cast(LanguageModel, torch.compile(model))
    return model


def build_loss(loss_key: str, device: torch.device) -> Loss:
    return LOSS_REGISTRY[loss_key]().to(device)


def build_callbacks(cfg: Config) -> list[Callback]:
    return [CALLBACK_REGISTRY[spec.type](**spec.params) for spec in cfg.callbacks]
