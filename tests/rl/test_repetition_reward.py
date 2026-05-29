"""Tests for the distinct-ngram variety reward (minichatbot/rl/rewards/repetition.py)."""

from __future__ import annotations

import pytest

from minichatbot.rl.rewards.repetition import DistinctNGramReward, _distinct_ratio


@pytest.mark.parametrize(
    ("tokens", "n", "expected"),
    [
        (["a", "b", "c"], 1, 1.0),  # all unigrams distinct
        (["a", "a", "a"], 1, 1 / 3),  # one distinct of three
        (["a", "b", "a", "b"], 2, 2 / 3),  # bigrams (a,b),(b,a),(a,b) -> 2 of 3
        (["a"], 2, 1.0),  # fewer than two n-grams -> 1.0
        ([], 1, 1.0),  # nothing to penalize
    ],
)
def test_distinct_ratio(tokens: list[str], n: int, expected: float) -> None:
    assert _distinct_ratio(tokens, n) == pytest.approx(expected)


def test_varied_text_scores_high() -> None:
    reward = DistinctNGramReward(min_words=4)
    assert reward("a b c d", "ref") == pytest.approx(1.0)


def test_repetitive_text_scores_low() -> None:
    reward = DistinctNGramReward(min_words=4)
    # distinct-1 = 1/4, distinct-2 = 1/3 -> variety 0.5*(0.25 + 0.333..)
    expected = 0.5 * (0.25 + 1 / 3)
    assert reward("a a a a", "ref") == pytest.approx(expected)


def test_empty_completion_scores_zero() -> None:
    assert DistinctNGramReward()("", "ref") == 0.0
    assert DistinctNGramReward()("!!! ???", "ref") == 0.0  # no word tokens


def test_length_floor_scales_short_completions() -> None:
    reward = DistinctNGramReward(min_words=8)
    # 4 distinct words, all-distinct variety 1.0, length_factor = 4/8 = 0.5.
    assert reward("a b c d", "ref") == pytest.approx(0.5)


def test_case_is_folded() -> None:
    reward = DistinctNGramReward(min_words=2)
    # "The the" -> one distinct unigram after lowercasing.
    assert reward("The the", "ref") == pytest.approx(0.5 * (0.5 + 1.0))


def test_reference_is_ignored() -> None:
    reward = DistinctNGramReward(min_words=4)
    a = reward("a b c d", "one reference")
    b = reward("a b c d", "a totally different reference")
    assert a == b


def test_metric_name_is_variety() -> None:
    assert DistinctNGramReward().metric_name == "mean variety"
