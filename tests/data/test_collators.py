"""Tests for batch collators (minichatbot/data/collators/*.py)."""

from __future__ import annotations

import torch

from minichatbot.data.collators.pretrain import PretrainCollator
from minichatbot.data.collators.rl import RLCollator
from minichatbot.data.collators.sft import SFTCollator


def test_pretrain_collator_shifts_and_stacks() -> None:
    # Each sample is a (seq_len + 1) chunk; input/labels are the shift pair.
    samples = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])]
    batch = PretrainCollator()(samples)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 3]))
    assert torch.equal(batch["labels"][0], torch.tensor([2, 3, 4]))


def test_sft_collator_right_pads_with_pad_and_ignore_index() -> None:
    pad_id = 99
    samples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3])},
        {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([-100, 5])},
    ]
    batch = SFTCollator(pad_id=pad_id)(samples)
    assert batch["input_ids"].shape == (2, 3)
    # Real region preserved; pad region filled with pad_id / -100.
    assert torch.equal(batch["input_ids"][1], torch.tensor([4, 5, pad_id]))
    assert torch.equal(batch["labels"][1], torch.tensor([-100, 5, -100]))


def test_sft_collator_from_config_uses_tokenizer_pad_id(stub_tokenizer) -> None:
    collator = SFTCollator.from_config(stub_tokenizer)
    assert collator.pad_id == stub_tokenizer.pad_id


def test_rl_collator_transposes_to_lists() -> None:
    # RL prompts aren't padded together; the collator just regroups the dicts.
    t0, t1 = torch.tensor([1, 2]), torch.tensor([3, 4, 5])
    samples = [
        {"prompt_ids": t0, "reference": "r0"},
        {"prompt_ids": t1, "reference": "r1"},
    ]
    batch = RLCollator()(samples)
    assert batch["prompt_ids"] == [t0, t1]
    assert batch["reference"] == ["r0", "r1"]
