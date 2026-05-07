"""Base class for autoregressive language models.

Concrete architectures (Transformer, RNN, Mamba, ...) implement this
interface so the trainer, inference, checkpointing, and registry
machinery treat them uniformly.

Save/load format (single .pt file):
    {
        "model_config": <ModelConfig as dict>,
        "model": <state_dict>,
        ... possibly more keys (e.g., from Trainer.save_checkpoint)
    }
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from minichatbot.config import ModelConfig
from minichatbot.utils.io import atomic_torch_save


@dataclass
class ModelOutput:
    logits: torch.Tensor
    state: Any | None = None


class LanguageModel(nn.Module, ABC):
    """Common interface for autoregressive language models.

    The `state` slot in forward / init_state is intentionally untyped so each
    architecture can use its own representation (per-layer KV tensors for
    transformers, per-layer hidden tensors for RNNs, etc.) while trainer and
    generator code stays architecture-agnostic.

    Subclasses MUST set `self.cfg: ModelConfig` in __init__ — `save()`,
    `load()`, and any orchestrator config introspection rely on it.
    """

    cfg: ModelConfig

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        state: Any | None = None,
    ) -> ModelOutput:
        """Forward pass.

        input_ids: (B, T) integer token ids
        state:     architecture-specific state from a previous step, or
                   None to start fresh (training, or first inference step).
        returns:   ModelOutput with logits (B, T, V) and optional new state.
        """

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> Any:
        """Build a fresh state object for stepwise inference."""

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> LanguageModel:
        """Build a model from config.

        - `LanguageModel.from_config(cfg)` — registry-dispatched: reads
          `cfg.type` and instantiates the matching architecture.
        - `Transformer.from_config(cfg)` (concrete subclass) — builds
          directly. Subclasses must override this method.
        """
        if cls is LanguageModel:
            # Lazy import to avoid the model/__init__.py <-> base.py cycle.
            from minichatbot.model import MODEL_REGISTRY

            target_cls = MODEL_REGISTRY[cfg.type]
            return target_cls.from_config(cfg)
        raise NotImplementedError(
            f"{cls.__name__} must override from_config"
        )

    def num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        """Save model weights + config to a single .pt file (atomic write)."""
        state = {
            "model_config": dataclasses.asdict(self.cfg),
            "model": self.state_dict(),
        }
        atomic_torch_save(state, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> LanguageModel:
        """Load a model from path. Dispatches by `cfg.type` via MODEL_REGISTRY.

        - `LanguageModel.load(path)` — registry-dispatched: reads
          `cfg.type` and instantiates the matching architecture. Use this
          when you don't know (or care) which architecture was saved.
        - `Transformer.load(path)` (concrete subclass) — validates that
          `cfg.type` matches the calling class and raises TypeError on
          mismatch. Useful when you want to assert the architecture.

        Either way the model is reconstructed via `from_config` then
        `load_state_dict`. The file may contain extra keys (e.g., a
        Trainer checkpoint) — they're ignored.
        """
        # Lazy import to avoid the model/__init__.py <-> base.py cycle.
        from minichatbot.model import MODEL_REGISTRY

        state = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**state["model_config"])
        registered = MODEL_REGISTRY[cfg.type]

        if cls is LanguageModel:
            target_cls: type[LanguageModel] = registered
        elif registered is not cls:
            raise TypeError(
                f"Checkpoint at {path} was saved as type={cfg.type!r} "
                f"({registered.__name__}); attempted to load as {cls.__name__}. "
                f"Use LanguageModel.load(path) for automatic dispatch, or load "
                f"with the matching concrete class."
            )
        else:
            target_cls = cls

        model = target_cls.from_config(cfg)
        model.load_state_dict(state["model"])
        return model
