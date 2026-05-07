"""Pretrain dataset over a packed uint16 token-id binary file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from minichatbot.config import DataConfig
from minichatbot.data import DATASET_REGISTRY
from minichatbot.data.base import BaseDataset
from minichatbot.tokenizer.base import Tokenizer


@DATASET_REGISTRY.register("pretrain")
class PretrainDataset(BaseDataset):
    """Random-access pretrain dataset over a flat uint16 token-id binary.

    Each `__getitem__(i)` returns a contiguous slice of length `seq_len + 1`
    starting at index `i`. The collator splits the slice into input_ids
    (`[:-1]`) and labels (`[1:]`). Document boundaries are encoded via EOS
    tokens that the data prep step appended; the model learns to predict
    EOS naturally.
    """

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.data) <= seq_len:
            raise ValueError(
                f"Data file {self.path} has {len(self.data)} tokens; "
                f"need > seq_len={seq_len} for at least one sample."
            )

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk)

    @classmethod
    def from_config(
        cls,
        cfg: DataConfig,
        tokenizer: Tokenizer,
        split: str = "train",
    ) -> PretrainDataset:
        if split == "train":
            path = cfg.train_path
        elif split == "val":
            if cfg.val_path is None:
                raise ValueError(
                    "PretrainDataset.from_config(split='val') requires "
                    "DataConfig.val_path to be set."
                )
            path = cfg.val_path
        else:
            raise ValueError(f"Unknown split: {split!r}")
        return cls(path=path, seq_len=cfg.seq_len)
