from minichatbot.training.losses.base import Loss
from minichatbot.utils.registry import Registry, import_submodules

LOSS_REGISTRY: Registry[Loss] = Registry("loss")

import_submodules(__name__, __path__)
