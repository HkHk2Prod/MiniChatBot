"""Tests for loglikelihood scoring (minichatbot/eval/scoring.py).

`_prepare` is pure and tested directly. `score_batch` is tested against a
StubLogitsModel whose logits are a fixed function of the current token, so
expected logprobs can be recomputed by hand with `log_softmax`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from _stubs import StubLogitsModel

from minichatbot.eval.scoring import _prepare, score_batch

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# _prepare — pure input construction / continuation accounting
# --------------------------------------------------------------------------- #


def test_prepare_prepends_prefix_for_empty_context() -> None:
    inp, n_cont = _prepare([], [5, 6], max_length=10, prefix_id=99)
    assert inp == [99, 5, 6]
    assert n_cont == 2


def test_prepare_left_truncates_keeping_continuation() -> None:
    # Oldest context tokens drop first; the continuation [6, 7] is retained.
    inp, n_cont = _prepare([1, 2, 3, 4, 5], [6, 7], max_length=4, prefix_id=0)
    assert inp == [4, 5, 6, 7]
    assert n_cont == 2


def test_prepare_caps_n_cont_at_len_minus_one() -> None:
    # Truncation leaves the first continuation token with no preceding token
    # to be predicted from, so only 2 of the 3 continuation tokens are scored.
    inp, n_cont = _prepare([1, 2], [3, 4, 5], max_length=3, prefix_id=0)
    assert inp == [3, 4, 5]
    assert n_cont == 2


def test_prepare_floors_n_cont_at_zero() -> None:
    inp, n_cont = _prepare([], [], max_length=10, prefix_id=7)
    assert inp == [7]
    assert n_cont == 0


# --------------------------------------------------------------------------- #
# score_batch — against a deterministic stub model
# --------------------------------------------------------------------------- #

# table[i] = next-token logits emitted when the current token is i.
#   token 0 -> uniform; token 1 -> favors 0; token 2 -> favors 1 (argmax = 1).
_TABLE = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ]
)


def _model() -> StubLogitsModel:
    return StubLogitsModel(_TABLE)


def _expected_logprob(prev_tokens: list[int], cont_tokens: list[int]) -> float:
    """Hand-compute summed logprob: cont token t predicted from prev_tokens[t]."""
    total = 0.0
    for prev, tok in zip(prev_tokens, cont_tokens, strict=True):
        total += float(F.log_softmax(_TABLE[prev], dim=-1)[tok].item())
    return total


def test_score_empty_continuation() -> None:
    results = score_batch(
        _model(), [([1], [])], device=CPU, max_length=10, prefix_id=0
    )
    assert results == [(0.0, True)]


def test_score_single_token_logprob_and_non_greedy() -> None:
    # inp = [1, 2]; cont token 2 is predicted from token 1's logits.
    (logprob, is_greedy), = score_batch(
        _model(), [([1], [2])], device=CPU, max_length=10, prefix_id=0
    )
    assert logprob == pytest.approx(_expected_logprob([1], [2]))
    # argmax of table[1] is token 0, not 2 -> not greedy.
    assert is_greedy is False


def test_score_is_greedy_true_when_continuation_is_argmax() -> None:
    # inp = [2, 1]; cont token 1 predicted from token 2's logits (argmax = 1).
    (logprob, is_greedy), = score_batch(
        _model(), [([2], [1])], device=CPU, max_length=10, prefix_id=0
    )
    assert logprob == pytest.approx(_expected_logprob([2], [1]))
    assert is_greedy is True


def test_score_multi_token_logprob_is_summed() -> None:
    # inp = [1, 2, 0]; cont [2, 0] predicted from tokens [1, 2] respectively.
    (logprob, is_greedy), = score_batch(
        _model(), [([1], [2, 0])], device=CPU, max_length=10, prefix_id=0
    )
    assert logprob == pytest.approx(_expected_logprob([1, 2], [2, 0]))
    assert is_greedy is False


def test_score_batch_preserves_order() -> None:
    batch = [([2], [1]), ([1], [2])]
    results = score_batch(_model(), batch, device=CPU, max_length=10, prefix_id=0)
    assert results[0][0] == pytest.approx(_expected_logprob([2], [1]))
    assert results[1][0] == pytest.approx(_expected_logprob([1], [2]))
    assert results[0][1] is True   # [2]->1 is greedy
    assert results[1][1] is False  # [1]->2 is not


def test_ragged_batch_matches_scoring_each_pair_alone() -> None:
    # Different lengths force right-padding; the causal/pad-invariant stub must
    # give the same scores batched as scored individually.
    batch = [([1], [2, 0]), ([2], [1])]
    batched = score_batch(_model(), batch, device=CPU, max_length=10, prefix_id=0)
    alone = [
        score_batch(_model(), [pair], device=CPU, max_length=10, prefix_id=0)[0]
        for pair in batch
    ]
    for got, want in zip(batched, alone, strict=True):
        assert got[0] == pytest.approx(want[0])
        assert got[1] == want[1]
