"""Tests for the pretrain and SFT datasets (minichatbot/data/{pretrain,sft}.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from minichatbot.chat.template import render_messages
from minichatbot.config import DataConfig
from minichatbot.data.pretrain import PretrainDataset
from minichatbot.data.sft import SFTDataset


def _write_bin(path: Path, values: list[int]) -> Path:
    np.array(values, dtype=np.uint16).tofile(path)
    return path


# --------------------------------------------------------------------------- #
# PretrainDataset
# --------------------------------------------------------------------------- #


def test_pretrain_dataset_slices_and_lengths(tmp_path: Path) -> None:
    path = _write_bin(tmp_path / "train.bin", [0, 1, 2, 3, 4, 5])
    ds = PretrainDataset(path, seq_len=3)
    assert len(ds) == 6 - 3  # one window per valid start index
    first = ds[0]
    assert first.dtype == torch.int64
    assert first.tolist() == [0, 1, 2, 3]   # length seq_len + 1
    assert ds[1].tolist() == [1, 2, 3, 4]


def test_pretrain_dataset_rejects_too_small_file(tmp_path: Path) -> None:
    path = _write_bin(tmp_path / "tiny.bin", [0, 1, 2])
    with pytest.raises(ValueError, match="need > seq_len"):
        PretrainDataset(path, seq_len=3)


def test_pretrain_from_config_splits(tmp_path: Path) -> None:
    train = _write_bin(tmp_path / "train.bin", [0, 1, 2, 3, 4])
    cfg = DataConfig(train_path=str(train), seq_len=2)
    assert len(PretrainDataset.from_config(cfg, tokenizer=None, split="train"))  # uses train_path

    with pytest.raises(ValueError, match="val_path"):
        PretrainDataset.from_config(cfg, tokenizer=None, split="val")
    with pytest.raises(ValueError, match="Unknown split"):
        PretrainDataset.from_config(cfg, tokenizer=None, split="test")


# --------------------------------------------------------------------------- #
# SFTDataset
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, conversations: list[list[dict]]) -> Path:
    lines = [json.dumps({"messages": c}) for c in conversations]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_CONVO = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "a detailed answer"},
]


def test_sft_dataset_loads_examples(tmp_path: Path, stub_tokenizer) -> None:
    path = _write_jsonl(tmp_path / "sft.jsonl", [_CONVO, _CONVO])
    ds = SFTDataset(path, tokenizer=stub_tokenizer, seq_len=100)
    assert len(ds) == 2
    item = ds[0]
    assert item["input_ids"].shape == item["labels"].shape
    assert any(label != -100 for label in item["labels"].tolist())  # assistant learned


def test_sft_dataset_truncates_to_seq_len(tmp_path: Path, stub_tokenizer) -> None:
    full_input, _ = render_messages(_CONVO, stub_tokenizer)
    seq_len = len(full_input) - 2  # truncate two tokens off the end
    path = _write_jsonl(tmp_path / "sft.jsonl", [_CONVO])
    ds = SFTDataset(path, tokenizer=stub_tokenizer, seq_len=seq_len)
    assert ds[0]["input_ids"].shape[0] == seq_len


def test_sft_dataset_drops_all_masked_after_truncation(tmp_path: Path, stub_tokenizer) -> None:
    # seq_len=2 keeps only the user-header prefix -> no learnable token -> dropped.
    path = _write_jsonl(tmp_path / "sft.jsonl", [_CONVO])
    ds = SFTDataset(path, tokenizer=stub_tokenizer, seq_len=2)
    assert len(ds) == 0


def test_sft_from_config_unknown_split_raises(tmp_path: Path, stub_tokenizer) -> None:
    path = _write_jsonl(tmp_path / "sft.jsonl", [_CONVO])
    cfg = DataConfig(train_path=str(path), seq_len=100)
    with pytest.raises(ValueError, match="Unknown split"):
        SFTDataset.from_config(cfg, tokenizer=stub_tokenizer, split="nope")
