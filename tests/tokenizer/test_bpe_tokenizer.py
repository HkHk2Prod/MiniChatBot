"""Tests for the byte-level BPE tokenizer (minichatbot/tokenizer/bpe.py).

Uses the session-scoped `tiny_bpe_tokenizer` fixture (a real tokenizer trained
on a small corpus). The byte-level pre-tokenizer adds a prefix space, so
decode reproduces the input with a single leading space — the tests account
for that while still asserting losslessness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minichatbot.tokenizer.bpe import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    BPETokenizer,
)

ROUND_TRIP_TEXTS = ["hello world", "the quick brown fox", "a b c", "1,000 apples"]


def test_vocab_includes_byte_alphabet_and_specials(
    tiny_bpe_tokenizer: BPETokenizer,
) -> None:
    # Byte-level alphabet is 256 entries; specials sit on top.
    assert tiny_bpe_tokenizer.vocab_size >= 256
    assert tiny_bpe_tokenizer.special_token_id(EOS_TOKEN) is not None
    assert tiny_bpe_tokenizer.special_token_id(PAD_TOKEN) is not None


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_encode_decode_preserves_content(tiny_bpe_tokenizer: BPETokenizer, text: str) -> None:
    ids = tiny_bpe_tokenizer.encode(text, include_special=False)
    decoded = tiny_bpe_tokenizer.decode(ids, include_special=False)
    # Byte-level adds a single prefix space; content is otherwise lossless.
    assert decoded == f" {text}"


def test_encode_decode_is_lossless_with_leading_space(
    tiny_bpe_tokenizer: BPETokenizer,
) -> None:
    text = " already has a leading space"
    ids = tiny_bpe_tokenizer.encode(text, include_special=False)
    assert tiny_bpe_tokenizer.decode(ids, include_special=False) == text


def test_include_special_appends_eos(tiny_bpe_tokenizer: BPETokenizer) -> None:
    with_special = tiny_bpe_tokenizer.encode("hello", include_special=True)
    without_special = tiny_bpe_tokenizer.encode("hello", include_special=False)
    assert with_special[-1] == tiny_bpe_tokenizer.eos_id
    assert with_special[:-1] == without_special


@pytest.mark.parametrize("include_special", [True, False])
def test_encode_batch_matches_per_item(
    tiny_bpe_tokenizer: BPETokenizer, include_special: bool
) -> None:
    texts = ROUND_TRIP_TEXTS
    batched = tiny_bpe_tokenizer.encode_batch(texts, include_special=include_special)
    per_item = [tiny_bpe_tokenizer.encode(t, include_special=include_special) for t in texts]
    assert batched == per_item


def test_special_token_id_lookups(tiny_bpe_tokenizer: BPETokenizer) -> None:
    assert tiny_bpe_tokenizer.special_token_id(EOS_TOKEN) == tiny_bpe_tokenizer.eos_id
    assert tiny_bpe_tokenizer.special_token_id(PAD_TOKEN) == tiny_bpe_tokenizer.pad_id
    assert tiny_bpe_tokenizer.special_token_id(BOS_TOKEN) == tiny_bpe_tokenizer.bos_id
    assert tiny_bpe_tokenizer.special_token_id("<|definitely-not-a-token|>") is None


def test_save_load_round_trip(tiny_bpe_tokenizer: BPETokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tiny_bpe_tokenizer.save(path)
    reloaded = BPETokenizer.load(path)

    text = "the quick brown fox"
    assert reloaded.encode(text) == tiny_bpe_tokenizer.encode(text)
    assert reloaded.vocab_size == tiny_bpe_tokenizer.vocab_size
    assert reloaded.eos_id == tiny_bpe_tokenizer.eos_id
