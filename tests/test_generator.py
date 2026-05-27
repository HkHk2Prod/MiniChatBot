"""Tests for the sampling Generator (minichatbot/inference/generator.py).

Greedy correctness (incl. KV-cache equivalence) is checked against the tiny
real Transformer; EOS early-stop and repetition penalties are checked against
a StubLogitsModel with hand-chosen logits for full control.
"""

from __future__ import annotations

import torch
from _stubs import StubLogitsModel

from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies.greedy import GreedySampling


def _manual_greedy(model, prompt: torch.Tensor, n: int) -> torch.Tensor:
    """Reference greedy decode using a full forward each step (no KV cache)."""
    ids = prompt.clone()
    with torch.no_grad():
        for _ in range(n):
            nxt = model(ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    return ids


def test_greedy_matches_full_forward_reference(tiny_transformer) -> None:
    # Generator uses the KV cache; the reference recomputes from scratch.
    # Equality proves both the cache and the generation loop are correct.
    prompt = torch.randint(0, tiny_transformer.cfg.vocab_size, (2, 4))
    gen = Generator(strategy=GreedySampling(), eos_id=None)
    got = gen.generate(tiny_transformer, prompt, max_new_tokens=6)
    assert torch.equal(got, _manual_greedy(tiny_transformer, prompt, 6))


def test_greedy_is_deterministic(tiny_transformer) -> None:
    prompt = torch.randint(0, tiny_transformer.cfg.vocab_size, (1, 3))
    gen = Generator(strategy=GreedySampling(), eos_id=None)
    a = gen.generate(tiny_transformer, prompt, max_new_tokens=5)
    b = gen.generate(tiny_transformer, prompt, max_new_tokens=5)
    assert torch.equal(a, b)


def test_output_length_without_eos() -> None:
    # Stub always favors token 0; with eos_id=None it never stops early.
    table = torch.tensor([[5.0, 0.0, 0.0]]).repeat(3, 1)
    model = StubLogitsModel(table)
    prompt = torch.tensor([[1, 2], [0, 1]])
    out = Generator(eos_id=None).generate(model, prompt, max_new_tokens=4)
    assert out.shape == (2, prompt.size(1) + 4)
    assert torch.all(out[:, prompt.size(1):] == 0)  # every new token is the argmax


def test_eos_early_stops() -> None:
    eos = 2
    table = torch.tensor([[0.0, 0.0, 10.0]]).repeat(3, 1)  # always favors token 2
    model = StubLogitsModel(table)
    prompt = torch.tensor([[0], [1]])
    out = Generator(eos_id=eos).generate(model, prompt, max_new_tokens=10)
    # Greedy emits EOS on the first step, so generation stops after one token.
    assert out.shape == (2, prompt.size(1) + 1)
    assert torch.all(out[:, -1] == eos)


def test_frequency_penalty_changes_output() -> None:
    # Without a penalty the stub repeats token 0 forever; a strong frequency
    # penalty should push generation off the repeated token.
    table = torch.tensor([[10.0, 9.0, 8.0]]).repeat(3, 1)
    model = StubLogitsModel(table)
    prompt = torch.tensor([[0]])

    plain = Generator(eos_id=None).generate(model, prompt, max_new_tokens=5)
    penalized = Generator(eos_id=None, frequency_penalty=2.0).generate(
        model, prompt, max_new_tokens=5
    )
    assert torch.all(plain[:, 1:] == 0)        # unpenalized: stuck on token 0
    assert not torch.equal(plain, penalized)   # penalty diversifies the output
