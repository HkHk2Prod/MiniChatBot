"""Tests for GRPO rollout collection (minichatbot/rl/rollout.py).

The model is a StubLogitsModel that greedily emits a fixed token, and rewards
come from a scripted sequence — so completions, advantages, masks, and padding
are all deterministic and hand-checkable without sampling noise.
"""

from __future__ import annotations

import torch
from _stubs import StubLogitsModel, StubTokenizer

from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies.greedy import GreedySampling
from minichatbot.rl.rewards.base import Reward
from minichatbot.rl.rollout import collect_rollouts
from minichatbot.tokenizer.bpe import IM_END_TOKEN

CPU = torch.device("cpu")
VOCAB = 5


def _const_model(token: int) -> StubLogitsModel:
    """A stub that greedily emits `token` at every step."""
    table = torch.full((VOCAB, VOCAB), 0.0)
    table[:, token] = 10.0
    return StubLogitsModel(table)


class _ScriptedReward(Reward):
    """Returns preset scores in call order, ignoring the completion text."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self._i = 0

    def __call__(self, completion: str, reference: str) -> float:
        v = self._scores[self._i]
        self._i += 1
        return float(v)


def _rollout(model, reward_fn, prompts, *, group_size, max_new_tokens, normalize_std):
    tok = StubTokenizer()
    chat_end = tok.special_token_id(IM_END_TOKEN)
    gen = Generator(strategy=GreedySampling(), eos_id=chat_end)
    return collect_rollouts(
        model=model,
        generator=gen,
        reward_fn=reward_fn,
        tokenizer=tok,
        prompts=prompts,
        references=["ref"] * len(prompts),
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        chat_end_id=chat_end,
        pad_id=tok.pad_id,
        device=CPU,
        normalize_advantage_std=normalize_std,
    )


def test_advantages_are_group_mean_centered() -> None:
    # One prompt, group of 4, rewards [1,0,0,0] -> advantage = reward - 0.25.
    res = _rollout(
        _const_model(3),
        _ScriptedReward([1, 0, 0, 0]),
        [torch.tensor([1, 2])],
        group_size=4,
        max_new_tokens=3,
        normalize_std=False,
    )
    assert res.batch["advantages"].tolist() == [0.75, -0.25, -0.25, -0.25]
    assert res.reward_mean == 0.25
    assert res.solve_rate == 0.25
    assert res.n_samples == 4


def test_advantages_std_normalized() -> None:
    res = _rollout(
        _const_model(3),
        _ScriptedReward([1, 0, 0, 0]),
        [torch.tensor([1, 2])],
        group_size=4,
        max_new_tokens=3,
        normalize_std=True,
    )
    group = torch.tensor([1.0, 0.0, 0.0, 0.0])
    expected = ((group - group.mean()) / (group.std(unbiased=False) + 1e-6)).tolist()
    got = res.batch["advantages"].tolist()
    for g, e in zip(got, expected, strict=True):
        assert abs(g - e) < 1e-5


def test_loss_mask_covers_completion_only() -> None:
    # prompt len 2, 3 generated tokens -> seq [1,2,3,3,3]; mask is 1 on the
    # 3 completion positions, 0 on the 2 prompt positions.
    res = _rollout(
        _const_model(3),
        _ScriptedReward([0.0]),
        [torch.tensor([1, 2])],
        group_size=1,
        max_new_tokens=3,
        normalize_std=False,
    )
    assert res.batch["input_ids"][0].tolist() == [1, 2, 3, 3, 3]
    assert res.batch["loss_mask"][0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_completion_trimmed_at_chat_end() -> None:
    tok = StubTokenizer()
    chat_end = tok.special_token_id(IM_END_TOKEN)
    res = _rollout(
        _const_model(chat_end),  # emits the chat-end token immediately
        _ScriptedReward([1.0]),
        [torch.tensor([1, 2])],
        group_size=1,
        max_new_tokens=8,
        normalize_std=False,
    )
    # Completion is trimmed to the single chat-end token (kept).
    assert res.batch["input_ids"][0].tolist() == [1, 2, chat_end]
    assert res.gen_len_mean == 1.0


def test_padding_to_max_length_with_pad_id() -> None:
    # Two prompts of different lengths -> rows padded to the longest seq.
    res = _rollout(
        _const_model(3),
        _ScriptedReward([0.0, 0.0]),
        [torch.tensor([1, 2]), torch.tensor([1, 2, 3])],
        group_size=1,
        max_new_tokens=3,
        normalize_std=False,
    )
    input_ids = res.batch["input_ids"]
    assert input_ids.shape == (2, 6)  # max seq = 3 prompt + 3 completion
    assert input_ids[0].tolist() == [1, 2, 3, 3, 3, 0]  # row 0 right-padded with pad_id
    assert res.batch["loss_mask"][0].tolist() == [0, 0, 1, 1, 1, 0]
