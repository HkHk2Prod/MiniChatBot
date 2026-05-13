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
    `rl.reward`. Build by key with `Reward.from_key(...)`.
    """

    @abstractmethod
    def __call__(self, completion: str, reference: str) -> float: ...

    @classmethod
    def from_key(cls, key: str) -> "Reward":
        # Lazy import to avoid the rewards/__init__.py <-> base.py cycle.
        from minichatbot.rl.rewards import REWARD_REGISTRY

        return REWARD_REGISTRY[key]()
