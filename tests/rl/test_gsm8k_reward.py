"""Tests for the GSM8K math reward (minichatbot/rl/rewards/gsm8k.py)."""

from __future__ import annotations

import pytest

from minichatbot.rl.rewards.gsm8k import (
    GSM8KReward,
    _normalize_number,
    extract_final_answer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,000", "1000"),  # thousands separators stripped
        ("1000", "1000"),
        ("1000.00", "1000"),  # trailing decimal zeros dropped -> integer
        ("1,000,000", "1000000"),
        ("-5", "-5"),  # negatives preserved
        ("3.14", "3.14"),  # genuine decimals kept
        ("-2.50", "-2.5"),  # negative + trailing zero trimmed
        ("0", "0"),
        ("abc", "abc"),  # non-numeric returned as-is
        ("", ""),
    ],
)
def test_normalize_number(raw: str, expected: str) -> None:
    assert _normalize_number(raw) == expected


def test_normalize_equates_thousands_and_decimal_forms() -> None:
    assert _normalize_number("1,000") == _normalize_number("1000") == _normalize_number("1000.00")


def test_extract_prefers_number_after_marker() -> None:
    # Numbers appear before the marker, but the one right after #### wins.
    text = "We add 3 and 4 to get 7.\n#### 42"
    assert extract_final_answer(text) == "42"


def test_extract_falls_back_to_last_number_without_marker() -> None:
    assert extract_final_answer("first 3, then 7, finally 11 apples") == "11"


def test_extract_marker_number_is_normalized() -> None:
    assert extract_final_answer("blah\n#### 1,000") == "1000"


def test_extract_returns_none_when_no_number() -> None:
    assert extract_final_answer("no numbers here at all") is None


def test_extract_marker_without_trailing_number_falls_back() -> None:
    # '####' present but nothing numeric after it -> fall back to last number.
    assert extract_final_answer("the total is 9 #### done") == "9"


@pytest.fixture
def reward() -> GSM8KReward:
    return GSM8KReward()


def test_reward_exact_match(reward: GSM8KReward) -> None:
    assert reward("The answer is #### 42", "#### 42") == 1.0


def test_reward_match_across_normalized_forms(reward: GSM8KReward) -> None:
    # completion writes "1000", reference uses "1,000" — both normalize equal.
    assert reward("so the total is 1000", "#### 1,000") == 1.0


def test_reward_mismatch(reward: GSM8KReward) -> None:
    assert reward("#### 41", "#### 42") == 0.0


def test_reward_zero_when_reference_has_no_number(reward: GSM8KReward) -> None:
    assert reward("#### 42", "no gold answer here") == 0.0


def test_reward_zero_when_completion_has_no_number(reward: GSM8KReward) -> None:
    assert reward("I am not sure", "#### 42") == 0.0


def test_reward_metric_name_is_solve_rate(reward: GSM8KReward) -> None:
    assert reward.metric_name == "solve_rate"


def test_reward_format_prediction_matches_extract(reward: GSM8KReward) -> None:
    completion = "after working it out, #### 100"
    assert reward.format_prediction(completion) == extract_final_answer(completion) == "100"
