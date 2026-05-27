"""Tests for the Transformer language model (minichatbot/model/transformer/model.py).

Uses the tiny (2-layer/16-dim) fixture so every forward is instant on CPU.
Focuses on architecture invariants that silently corrupt training when broken:
causality, KV-cache equivalence, weight tying, and save/load fidelity.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minichatbot.config import ModelConfig
from minichatbot.model import MODEL_REGISTRY
from minichatbot.model.base import LanguageModel
from minichatbot.model.transformer.model import Transformer

# A second registered architecture so we can exercise the cross-type load guard.
# Abstract (never instantiated) — `load` raises before construction on a mismatch.
if "dummy_arch" not in MODEL_REGISTRY:

    @MODEL_REGISTRY.register("dummy_arch")
    class _DummyArch(LanguageModel):
        pass


def test_forward_output_shape(tiny_transformer: Transformer) -> None:
    ids = torch.randint(0, tiny_transformer.cfg.vocab_size, (3, 7))
    out = tiny_transformer(ids)
    assert out.logits.shape == (3, 7, tiny_transformer.cfg.vocab_size)


def test_forward_is_causal(tiny_transformer: Transformer) -> None:
    # Changing only the LAST token must not affect logits at earlier positions.
    ids = torch.randint(0, tiny_transformer.cfg.vocab_size, (2, 6))
    other = ids.clone()
    other[:, -1] = (other[:, -1] + 1) % tiny_transformer.cfg.vocab_size
    with torch.no_grad():
        a = tiny_transformer(ids).logits
        b = tiny_transformer(other).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_tie_embeddings_shares_weight(tiny_transformer: Transformer) -> None:
    assert tiny_transformer.lm_head.weight is tiny_transformer.tok_embed.weight


def test_untied_embeddings_are_distinct(tiny_model_config: ModelConfig) -> None:
    cfg = ModelConfig(**{**tiny_model_config.__dict__, "tie_embeddings": False})
    model = Transformer(cfg)
    assert model.lm_head.weight is not model.tok_embed.weight


def test_forward_exceeding_max_seq_len_raises(tiny_transformer: Transformer) -> None:
    too_long = tiny_transformer.cfg.max_seq_len + 1
    ids = torch.randint(0, tiny_transformer.cfg.vocab_size, (1, too_long))
    with pytest.raises(ValueError, match="exceeds"):
        tiny_transformer(ids)


def test_kv_cache_matches_full_forward(tiny_transformer: Transformer) -> None:
    # Decoding token-by-token through the KV cache must reproduce the logits
    # of a single full-sequence forward at each position.
    ids = torch.randint(0, tiny_transformer.cfg.vocab_size, (2, 9))
    with torch.no_grad():
        full = tiny_transformer(ids).logits
        state = tiny_transformer.init_state(2, torch.device("cpu"))
        for i in range(ids.size(1)):
            out = tiny_transformer(ids[:, i : i + 1], state=state)
            state = out.state
            assert torch.allclose(out.logits[:, -1], full[:, i], atol=1e-5)


def test_init_state_has_empty_cache_per_layer(tiny_transformer: Transformer) -> None:
    state = tiny_transformer.init_state(4, torch.device("cpu"))
    assert len(state) == tiny_transformer.cfg.n_layers
    for cache in state:
        # (batch, heads, seq=0, head_dim)
        assert cache.k.shape == (4, tiny_transformer.n_heads, 0, tiny_transformer.head_dim)


def test_num_params_counts_parameters(tiny_transformer: Transformer) -> None:
    expected = sum(p.numel() for p in tiny_transformer.parameters() if p.requires_grad)
    assert tiny_transformer.num_params() == expected


def test_save_load_round_trip_via_base(
    tiny_transformer: Transformer, tmp_path: Path
) -> None:
    path = tmp_path / "model.pt"
    tiny_transformer.save(path)
    ids = torch.randint(0, tiny_transformer.cfg.vocab_size, (2, 5))

    for loaded in (LanguageModel.load(path), Transformer.load(path)):
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(loaded(ids).logits, tiny_transformer(ids).logits, atol=1e-6)


def test_load_as_wrong_concrete_class_raises_type_error(
    tiny_transformer: Transformer, tmp_path: Path
) -> None:
    path = tmp_path / "model.pt"
    tiny_transformer.save(path)  # saved with type="transformer"
    DummyArch = MODEL_REGISTRY["dummy_arch"]
    with pytest.raises(TypeError, match="saved as type"):
        DummyArch.load(path)
