"""SFT dataset over a JSONL file of {"messages": [...]} rows.

Each JSONL line is a complete conversation. `render_messages` produces
(input_ids, labels) per row; the SFTCollator pads to the batch's max
length. Conversations are pre-tokenized at __init__ to keep __getitem__
allocation-free during training.

For very large SFT corpora (>>1M examples) consider streaming, but
typical SFT runs fit comfortably in RAM — alpaca-cleaned is ~50K rows,
LIMA is ~1K, even ShareGPT is well under a million. Eager loading is
the right default.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from minichatbot.chat.template import render_messages
from minichatbot.config import DataConfig
from minichatbot.data import DATASET_REGISTRY
from minichatbot.data.base import BaseDataset
from minichatbot.tokenizer.base import Tokenizer


@DATASET_REGISTRY.register("sft")
class SFTDataset(BaseDataset):
    """Random-access SFT dataset.

    Each item is `{"input_ids": LongTensor, "labels": LongTensor}` where
    `labels` is shifted-and-masked per `render_messages`. Sequences are
    truncated to `seq_len` tokens; if the assistant response sits past
    that boundary the sample is silently dropped (loss with all-(-100)
    labels would NaN). A warning prints how many were dropped.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples: list[tuple[list[int], list[int]]] = []
        self._load()

    def _load(self) -> None:
        n_total = 0
        n_dropped = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                obj = json.loads(line)
                messages = obj["messages"]
                input_ids, labels = render_messages(messages, self.tokenizer)
                # Truncate from the right (preserves system + user prefix; loss
                # tokens that fall off are simply not learned this epoch).
                if len(input_ids) > self.seq_len:
                    input_ids = input_ids[: self.seq_len]
                    labels = labels[: self.seq_len]
                # Drop examples with no learnable position after truncation —
                # they'd produce a NaN loss (all targets ignored).
                if not any(t != -100 for t in labels):
                    n_dropped += 1
                    continue
                self.examples.append((input_ids, labels))
        if n_dropped:
            print(
                f"[sft-dataset] dropped {n_dropped}/{n_total} examples with no "
                f"learnable tokens after truncation (seq_len={self.seq_len})."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self.examples[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    @classmethod
    def from_config(
        cls,
        cfg: DataConfig,
        tokenizer: Tokenizer,
        split: str = "train",
    ) -> SFTDataset:
        if split == "train":
            path = cfg.train_path
        elif split == "val":
            if cfg.val_path is None:
                raise ValueError(
                    "SFTDataset.from_config(split='val') requires "
                    "DataConfig.val_path to be set."
                )
            path = cfg.val_path
        else:
            raise ValueError(f"Unknown split: {split!r}")
        return cls(path=path, tokenizer=tokenizer, seq_len=cfg.seq_len)
