"""Reward for GSM8K-style grade-school math: 1.0 iff the final number matches.

GSM8K reference answers end with a `#### <answer>` marker; we prefer the
number right after that marker, falling back to the last number anywhere
in the string (handles model completions that don't follow the format).
Numbers are normalized before comparison — thousands separators stripped,
trailing decimal zeros dropped — so "1,000", "1000", and "1000.00" all
compare equal.

This is the canonical nanochat-style RL reward: a sparse, verifiable
binary signal. Shaped variants (partial credit, format bonuses) belong
in their own `Reward` subclasses.
"""

from __future__ import annotations

import re

from minichatbot.rl.rewards import REWARD_REGISTRY
from minichatbot.rl.rewards.base import Reward

# A run of digits, optionally with thousands separators / commas and a
# decimal part, optionally signed. Deliberately permissive.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(raw: str) -> str:
    cleaned = raw.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value.is_integer():
        return str(int(value))
    return repr(value)


def extract_final_answer(text: str) -> str | None:
    """Pull the final numeric answer out of a GSM8K answer or a completion.

    Returns the normalized number string, or None if no number is found.
    """
    if "####" in text:
        after = text.rsplit("####", 1)[1]
        m = _NUMBER_RE.search(after)
        if m is not None:
            return _normalize_number(m.group())
    matches = _NUMBER_RE.findall(text)
    if matches:
        return _normalize_number(matches[-1])
    return None


@REWARD_REGISTRY.register("gsm8k")
class GSM8KReward(Reward):
    """1.0 if the completion's final number equals the reference's, else 0.0."""

    def __call__(self, completion: str, reference: str) -> float:
        gold = extract_final_answer(reference)
        if gold is None:
            return 0.0
        pred = extract_final_answer(completion)
        return 1.0 if pred is not None and pred == gold else 0.0
