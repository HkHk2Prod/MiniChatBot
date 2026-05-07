"""Optimizer and learning-rate scheduler builders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from minichatbot.config import OptimConfig


def _split_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split params into decay (matrices, embeddings) and no-decay (biases, norms).

    Convention: anything 1-D (biases, layernorm/RMSNorm gain) is no-decay;
    anything 2-D+ (linear weights, embeddings) gets weight decay. Standard
    in nanoGPT / GPT-NeoX / LLaMA training scripts.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (no_decay if p.ndim < 2 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        _split_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
    )


def _cosine_lambda(warmup: int, total: int, min_ratio: float):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(progress, 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return fn


def _linear_lambda(warmup: int, total: int, min_ratio: float):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(progress, 1.0)
        return max(min_ratio, 1.0 - (1.0 - min_ratio) * progress)

    return fn


def _constant_lambda(warmup: int):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        return 1.0

    return fn


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimConfig,
    max_steps: int,
) -> LambdaLR:
    if cfg.lr_schedule == "cosine":
        fn = _cosine_lambda(cfg.warmup_steps, max_steps, cfg.min_lr_ratio)
    elif cfg.lr_schedule == "linear":
        fn = _linear_lambda(cfg.warmup_steps, max_steps, cfg.min_lr_ratio)
    elif cfg.lr_schedule == "constant":
        fn = _constant_lambda(cfg.warmup_steps)
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")
    return LambdaLR(optimizer, lr_lambda=fn)
