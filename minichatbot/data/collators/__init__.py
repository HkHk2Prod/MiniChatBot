from minichatbot.data.collators.base import Collator
from minichatbot.utils.registry import Registry, import_submodules

COLLATOR_REGISTRY: Registry[Collator] = Registry("collator")

import_submodules(__name__, __path__)
