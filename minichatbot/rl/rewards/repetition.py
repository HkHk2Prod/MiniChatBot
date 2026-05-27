"""Reward for lexical variety: penalizes the degenerate repetition loops
small language models fall into ("and ran and ran and ran").

The score is the mean of the distinct-1 and distinct-2 ratios — the
fraction of *unique* unigrams and *unique* bigrams in the completion. A
varied, coherent story scores near 1.0; a policy stuck repeating a token
or a phrase scores near 0.0. Both orders earn their keep: distinct-1
catches single-token loops ("the the the"), distinct-2 catches phrase
loops ("and ran and ran") that distinct-1 alone would miss.

A length floor closes the obvious exploit — a two-word completion is
trivially "all distinct". Below `min_words` the score is scaled down
linearly, so the policy can't win by collapsing toward empty output. The
floor is a floor only: completions at or above it are never penalized for
length, keeping this a pure *variety* signal (length *control* is a
separate reward).

This is reference-free — it reads only the completion, so the dataset's
`answer` field is ignored (use any placeholder). It pairs with an
SFT-initialized policy and a small LR: starting from a coherent model and
nudging gently, "more distinct" reliably means "less repetitive" rather
than "more novel gibberish". Like every reward here it feeds GRPO, which
uses only within-group *differences*, so the absolute scale is irrelevant
— what matters is that the looped sample in a group scores below its
varied siblings. Registered as `distinct_ngram`.
"""

from __future__ import annotations

import re

from minichatbot.rl.rewards import REWARD_REGISTRY
from minichatbot.rl.rewards.base import Reward

# Word-ish runs: letters/digits/underscore. Deliberately simple — we want
# a cheap, language-agnostic token stream, not linguistically correct
# tokenization. Casing is folded so "The" and "the" count as one word.
_WORD_RE = re.compile(r"\w+")


def _distinct_ratio(tokens: list[str], n: int) -> float:
    """Fraction of distinct n-grams among all n-grams in `tokens`.

    Returns 1.0 when there are fewer than two n-grams: a single n-gram
    cannot repeat, so there is nothing to penalize (the length floor in
    the reward handles "too short to be meaningful").
    """
    if len(tokens) < n + 1:
        return 1.0
    ngrams = list(zip(*(tokens[i:] for i in range(n)), strict=False))
    return len(set(ngrams)) / len(ngrams)


@REWARD_REGISTRY.register("distinct_ngram")
class DistinctNGramReward(Reward):
    """Mean of distinct-1 and distinct-2 ratios, scaled by a length floor."""

    # Continuous [0, 1] variety score — "solve_rate" would be a misnomer.
    metric_name = "mean variety"

    def __init__(self, min_words: int = 12) -> None:
        self.min_words = min_words

    def __call__(self, completion: str, reference: str) -> float:
        tokens = [t.lower() for t in _WORD_RE.findall(completion)]
        if not tokens:
            return 0.0
        variety = 0.5 * (_distinct_ratio(tokens, 1) + _distinct_ratio(tokens, 2))
        length_factor = min(1.0, len(tokens) / self.min_words)
        return variety * length_factor
