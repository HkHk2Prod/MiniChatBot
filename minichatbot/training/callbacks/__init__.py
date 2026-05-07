from minichatbot.training.callbacks.base import Callback, CallbackContext
from minichatbot.utils.registry import Registry, import_submodules

CALLBACK_REGISTRY: Registry[Callback] = Registry("callback")

import_submodules(__name__, __path__)
