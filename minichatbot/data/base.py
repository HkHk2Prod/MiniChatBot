"""Base class for stage-specific datasets."""

from __future__ import annotations

from abc import ABC
from typing import Any

from torch.utils.data import Dataset

from minichatbot.config import DataConfig
from minichatbot.tokenizer.base import Tokenizer


class BaseDataset(Dataset[Any], ABC):
    """Common factory interface for all training datasets.

    Inherits from `torch.utils.data.Dataset` so any subclass is directly
    usable with `DataLoader`. PyTorch's `IterableDataset` is itself a
    `Dataset` subclass, so streaming datasets are still allowed —
    subclass `IterableDataset` (which already satisfies `Dataset`) and
    this `BaseDataset` interface side-by-side.
    """

    @classmethod
    def from_config(
        cls,
        cfg: DataConfig,
        tokenizer: Tokenizer,
        split: str = "train",
    ) -> BaseDataset:
        """Build a dataset for the given split ('train' or 'val').

        - `BaseDataset.from_config(cfg, tok)` — registry-dispatched: reads
          `cfg.type` and instantiates the matching dataset class. Use this
          when you want to stay decoupled from concrete classes.
        - `PretrainDataset.from_config(cfg, tok)` (concrete subclass) —
          builds directly. Subclasses must override this method.

        Subclasses should raise on unsupported splits rather than fall
        back silently — surfaces mis-configured runs early.
        """
        if cls is BaseDataset:
            # Lazy import to avoid base.py <-> __init__.py cycle.
            from minichatbot.data import DATASET_REGISTRY

            target_cls = DATASET_REGISTRY[cfg.type]
            return target_cls.from_config(cfg, tokenizer, split)
        raise NotImplementedError(
            f"{cls.__name__} must override from_config"
        )
