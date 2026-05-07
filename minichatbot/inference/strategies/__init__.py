from minichatbot.inference.strategies.base import SamplingStrategy
from minichatbot.utils.registry import Registry, import_submodules

SAMPLING_REGISTRY: Registry[SamplingStrategy] = Registry("sampling_strategy")

import_submodules(__name__, __path__)
