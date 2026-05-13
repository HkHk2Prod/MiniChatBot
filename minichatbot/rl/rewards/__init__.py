from minichatbot.rl.rewards.base import Reward
from minichatbot.utils.registry import Registry, import_submodules

REWARD_REGISTRY: Registry[Reward] = Registry("reward")

import_submodules(__name__, __path__)
