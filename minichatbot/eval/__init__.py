"""Standardized evaluation glue.

`scoring.py` is the dependency-free core: turn (context, continuation)
token pairs into log-probabilities using a `LanguageModel`'s forward
pass. `lm_eval_adapter.py` wraps that core in the `LM` interface that
EleutherAI's lm-evaluation-harness drives, so the harness can score our
model on its standard task suite (lambada, piqa, arc, hellaswag,
wikitext, gsm8k, ...). The adapter import is intentionally NOT re-exported
here so `import minichatbot.eval.scoring` works without `lm-eval` installed.
"""

from minichatbot.eval.scoring import score_batch

__all__ = ["score_batch"]
