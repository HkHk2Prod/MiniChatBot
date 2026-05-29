"""Tests for the text-level generation wrapper (minichatbot/inference/text_generator.py).

Pairs a real tiny Transformer (sized to the tiny BPE tokenizer's vocab) with a
greedy Generator so encode -> generate -> decode is deterministic.
"""

from __future__ import annotations

import pytest
import torch

from minichatbot.config import ModelConfig
from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies.greedy import GreedySampling
from minichatbot.inference.text_generator import TextGenerator
from minichatbot.model.transformer.model import Transformer


@pytest.fixture
def text_generator(tiny_bpe_tokenizer):
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=tiny_bpe_tokenizer.vocab_size,
        max_seq_len=64,
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    model = Transformer(cfg)
    model.eval()
    gen = Generator(strategy=GreedySampling(), eos_id=None)
    return TextGenerator(model, tiny_bpe_tokenizer, gen)


def test_single_prompt_returns_str(text_generator) -> None:
    out = text_generator.generate("hello", max_new_tokens=3)
    assert isinstance(out, str)


def test_list_prompt_returns_list(text_generator) -> None:
    out = text_generator.generate(["hello", "world"], max_new_tokens=3)
    assert isinstance(out, list)
    assert len(out) == 2 and all(isinstance(s, str) for s in out)


def test_empty_list_returns_empty_list(text_generator) -> None:
    assert text_generator.generate([], max_new_tokens=3) == []


def test_return_only_completion_matches_manual_decode(text_generator) -> None:
    prompt = "hello"
    got = text_generator.generate(prompt, max_new_tokens=4, return_only_completion=True)

    ids = text_generator.tokenizer.encode(prompt, include_special=False)
    out = Generator(strategy=GreedySampling(), eos_id=None).generate(
        text_generator.model, torch.tensor([ids]), max_new_tokens=4
    )
    expected = text_generator.tokenizer.decode(out[0].tolist()[len(ids) :], include_special=False)
    assert got == expected


def test_full_output_is_at_least_completion(text_generator) -> None:
    only = text_generator.generate("hello", max_new_tokens=4, return_only_completion=True)
    full = text_generator.generate("hello", max_new_tokens=4, return_only_completion=False)
    assert len(full) >= len(only)  # full includes the prompt prefix


def test_generate_chat_returns_reply_str(text_generator) -> None:
    reply = text_generator.generate_chat("hi there", max_new_tokens=3)
    assert isinstance(reply, str)
