"""Shared pytest fixtures.

Kept lightweight: a tiny BPE tokenizer trained once per session, plus the
char-level / fixed-logit stubs from ``_stubs`` for the modules that only need
a Tokenizer or LanguageModel *interface*, not a real one.
"""

from __future__ import annotations

import pytest
import torch
from _stubs import StubLogitsModel, StubTokenizer

from minichatbot.config import ModelConfig
from minichatbot.model.transformer.model import Transformer
from minichatbot.tokenizer.bpe import BPETokenizer

__all__ = ["StubLogitsModel", "StubTokenizer"]

# A handful of short, overlapping lines — enough for the BPE trainer to learn
# a few merges on top of the byte-level alphabet without being slow.
_TINY_CORPUS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "hello there, world",
    "the cat sat on the mat",
    "a b c d e f g h i j k l m n o p",
    "numbers like 1, 2, 3 and 1,000 appear too",
] * 16


@pytest.fixture(scope="session")
def tiny_bpe_tokenizer() -> BPETokenizer:
    """A real BPETokenizer trained on a tiny corpus (session-scoped).

    ``vocab_size=320`` leaves room above the 256-entry byte alphabet plus the
    default special tokens for a few learned merges. ``min_frequency=1`` so the
    small corpus still produces merges.
    """
    return BPETokenizer.train(
        _TINY_CORPUS,
        vocab_size=320,
        min_frequency=1,
        show_progress=False,
    )


@pytest.fixture
def stub_tokenizer() -> StubTokenizer:
    """A char-level tokenizer with the ChatML special tokens present."""
    return StubTokenizer()


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    """A 2-layer/16-dim transformer config — instant to build and run on CPU."""
    return ModelConfig(
        vocab_size=32,
        max_seq_len=32,
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
        tie_embeddings=True,
        norm_type="rmsnorm",
    )


@pytest.fixture
def tiny_transformer(tiny_model_config: ModelConfig) -> Transformer:
    """A seeded tiny Transformer in eval mode (deterministic forward)."""
    torch.manual_seed(0)
    model = Transformer(tiny_model_config)
    model.eval()
    return model
