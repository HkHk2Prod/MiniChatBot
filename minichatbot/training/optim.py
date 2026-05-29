"""Optimizer and learning-rate scheduler builders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from minichatbot.config import OptimConfig
from minichatbot.utils.torch_helpers import unwrap_compiled

# config name -> bitsandbytes optimizer class name (lazily resolved).
_BNB_OPTIMIZERS = {"adamw_8bit": "AdamW8bit", "paged_adamw_8bit": "PagedAdamW8bit"}


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
    param_groups = _split_param_groups(model, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    return _build_bnb_8bit(model, param_groups, cfg)


def _build_bnb_8bit(
    model: nn.Module, param_groups: list[dict], cfg: OptimConfig
) -> torch.optim.Optimizer:
    """bitsandbytes 8-bit AdamW (optionally paged). Quantizes the Adam moment
    buffers to ~8-bit — a large cut to the fp32 optimizer state — while keeping
    embedding moments in fp32, which is bitsandbytes' documented recipe (8-bit
    state on the big, sparse-gradient embedding table degrades quality)."""
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            f"optim.optimizer={cfg.optimizer!r} needs bitsandbytes (CUDA-only). "
            'Install with: pip install -e ".[bnb]"'
        ) from e

    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    if device.type != "cuda":
        raise RuntimeError(
            f"optim.optimizer={cfg.optimizer!r} (bitsandbytes 8-bit) requires a CUDA "
            f"device, but the model is on {device.type!r}. Use optimizer=adamw instead."
        )

    # Pin embedding moments to fp32. The manager matches by parameter, so the
    # tied lm_head weight (same tensor) is covered too.
    manager = bnb.optim.GlobalOptimManager.get_instance()
    for module in unwrap_compiled(model).modules():
        if isinstance(module, nn.Embedding):
            manager.register_module_override(module, "weight", {"optim_bits": 32})

    optim_cls = getattr(bnb.optim, _BNB_OPTIMIZERS[cfg.optimizer])
    return optim_cls(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)


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
    match cfg.lr_schedule:
        case "cosine":
            fn = _cosine_lambda(cfg.warmup_steps, max_steps, cfg.min_lr_ratio)
        case "linear":
            fn = _linear_lambda(cfg.warmup_steps, max_steps, cfg.min_lr_ratio)
        case "constant":
            fn = _constant_lambda(cfg.warmup_steps)
        case _:
            raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")
    return LambdaLR(optimizer, lr_lambda=fn)
