"""Reinforcement-learning stage (GRPO).

The RL stage takes an SFT checkpoint and improves it against a *reward
function* instead of a labelled target: for each prompt it samples a
group of completions, scores each with a `Reward`, mean-centers the
scores within the group to form advantages, and takes one on-policy
policy-gradient step (see `minichatbot.training.rl_trainer.GRPOTrainer`).

Layout:
    rewards/        Reward implementations (REWARD_REGISTRY)
    rollout.py      collect_rollouts() — sampling + reward + advantage + batching
"""

from minichatbot.rl.rewards import REWARD_REGISTRY
from minichatbot.rl.rewards.base import Reward
from minichatbot.rl.rollout import RolloutResult, collect_rollouts

__all__ = ["REWARD_REGISTRY", "Reward", "RolloutResult", "collect_rollouts"]
