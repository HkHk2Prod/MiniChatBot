from minichatbot.model.base import LanguageModel, ModelOutput
from minichatbot.utils.registry import Registry, import_submodules

MODEL_REGISTRY: Registry[LanguageModel] = Registry("model")

import_submodules(__name__, __path__)
