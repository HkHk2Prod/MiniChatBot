"""Base class for RL reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Reward(ABC):
    """Scores one sampled completion against its reference answer.

    The return value can be on any scale — GRPO only uses within-group
    *differences* (advantages), so a binary 0/1 correctness signal is as
    valid as a shaped reward. Implementations should be pure functions of
    `(completion, reference)` and cheap to call (they run once per
    sampled completion, i.e. `group_size` times per prompt per step).

    Concrete rewards register under a string key; configs select one via
    `rl.reward`. Look up by key with `REWARD_REGISTRY[key]()`.

    The two display hooks below are only consulted by the benchmark
    engine (`minichatbot.inference.benchmark`) when it renders a scored
    run; they have no effect on training. Override them to make the
    benchmark output speak the reward's own language instead of GSM8K's.
    """

    #: Label the benchmark summary uses for the per-group mean score.
    #: "solve_rate" reads right for a binary correctness reward; a
    #: continuous reward should pick something like "mean variety".
    metric_name: str = "mean score"

    @abstractmethod
    def __call__(self, completion: str, reference: str) -> float: ...

    def format_prediction(self, completion: str) -> str | None:
        """Short, human-readable extraction shown as `PRED:` in the
        benchmark dump (e.g. GSM8K's final number). Return None — the
        default — to omit the line for rewards with nothing to extract.
        """
        return None
