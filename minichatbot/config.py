"""Typed configuration schema for MiniChatBot.

Every component (model, tokenizer, data, optim, trainer, callbacks) consumes a
dataclass that mirrors a section of the YAML config. `load_config(path)` parses
and validates a YAML file; `save_config(cfg, path)` snapshots a config to disk
so each run records the exact settings it was launched with.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    type: str = "transformer"
    vocab_size: int = 32000
    max_seq_len: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_type: str = "rmsnorm"


@dataclass
class TokenizerConfig:
    type: str = "bpe"
    path: str = "tokenizer.json"
    vocab_size: int | None = None


@dataclass
class DataConfig:
    train_path: str
    type: str = "pretrain"
    val_path: str | None = None
    seq_len: int = 1024
    num_workers: int = 4
    # RL only: optional system prompt prepended to every sampled prompt.
    # Ignored by the pretrain/SFT datasets.
    system_prompt: str | None = None


@dataclass
class OptimConfig:
    lr: float = 3.0e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1


@dataclass
class TrainerConfig:
    max_steps: int = 10000
    batch_size: int = 16
    grad_accum_steps: int = 1
    grad_clip: float | None = 1.0
    precision: str = "bf16"
    compile: bool = False


@dataclass
class RLConfig:
    """GRPO reinforcement-learning stage knobs.

    Only consumed by `scripts/train/rl.py`; pretrain/SFT ignore it.
    `group_size` completions are sampled per prompt and scored by the
    reward function named in `reward` (see `minichatbot.rl.REWARD_REGISTRY`);
    advantages are computed by mean-centering each group (and dividing by
    the group std when `normalize_advantage_std` is set — the "GR" in GRPO).
    """

    group_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    normalize_advantage_std: bool = True
    reward: str = "gsm8k"


@dataclass
class CallbackSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Config:
    run_name: str
    data: DataConfig
    output_dir: str = "runs"
    seed: int = 42
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    callbacks: list[CallbackSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return load_config(path)

    def to_yaml(self, path: str | Path) -> None:
        save_config(self, path)


def _from_dict(tp: Any, data: Any) -> Any:
    if data is None:
        return None

    origin = typing.get_origin(tp)

    if origin is typing.Union:
        non_none = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _from_dict(non_none[0], data)
        return data

    if origin is list:
        (item_type,) = typing.get_args(tp)
        return [_from_dict(item_type, x) for x in data]

    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_dict(args[0], x) for x in data)
        return tuple(_from_dict(t, x) for t, x in zip(args, data, strict=True))

    if origin is dict:
        return dict(data)

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        valid = {f.name for f in dataclasses.fields(tp)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown fields for {tp.__name__}: {sorted(unknown)}. "
                f"Valid fields: {sorted(valid)}."
            )
        kwargs = {name: _from_dict(hints[name], data[name]) for name in data}
        return tp(**kwargs)

    return data


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root at {path} must be a YAML mapping, got {type(raw).__name__}"
        )
    cfg = _from_dict(Config, raw)
    validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)


_VALID_PRECISION = {"fp32", "fp16", "bf16"}
_VALID_NORM = {"rmsnorm", "layernorm"}
_VALID_LR_SCHED = {"cosine", "linear", "constant"}
_VALID_DEVICE = {"auto", "cuda", "cpu", "mps"}


def validate(cfg: Config) -> None:
    if cfg.trainer.precision not in _VALID_PRECISION:
        raise ValueError(
            f"trainer.precision={cfg.trainer.precision!r}; expected one of {sorted(_VALID_PRECISION)}"
        )
    if cfg.model.norm_type not in _VALID_NORM:
        raise ValueError(
            f"model.norm_type={cfg.model.norm_type!r}; expected one of {sorted(_VALID_NORM)}"
        )
    if cfg.optim.lr_schedule not in _VALID_LR_SCHED:
        raise ValueError(
            f"optim.lr_schedule={cfg.optim.lr_schedule!r}; expected one of {sorted(_VALID_LR_SCHED)}"
        )
    if cfg.device not in _VALID_DEVICE:
        raise ValueError(
            f"device={cfg.device!r}; expected one of {sorted(_VALID_DEVICE)}"
        )
    if cfg.model.d_model % cfg.model.n_heads != 0:
        raise ValueError(
            f"model.d_model ({cfg.model.d_model}) must be divisible by "
            f"model.n_heads ({cfg.model.n_heads})"
        )
    if cfg.trainer.batch_size < 1:
        raise ValueError(f"trainer.batch_size must be >= 1, got {cfg.trainer.batch_size}")
    if cfg.trainer.grad_accum_steps < 1:
        raise ValueError(
            f"trainer.grad_accum_steps must be >= 1, got {cfg.trainer.grad_accum_steps}"
        )
    if cfg.data.seq_len > cfg.model.max_seq_len:
        raise ValueError(
            f"data.seq_len ({cfg.data.seq_len}) cannot exceed "
            f"model.max_seq_len ({cfg.model.max_seq_len})"
        )
    if cfg.rl.group_size < 2:
        raise ValueError(
            f"rl.group_size must be >= 2 (need a group to baseline against), "
            f"got {cfg.rl.group_size}"
        )
    if not 0.0 < cfg.rl.top_p <= 1.0:
        raise ValueError(f"rl.top_p must be in (0, 1], got {cfg.rl.top_p}")
