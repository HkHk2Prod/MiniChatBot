from minichatbot.tokenizer.base import Tokenizer
from minichatbot.utils.registry import Registry, import_submodules

TOKENIZER_REGISTRY: Registry[Tokenizer] = Registry("tokenizer")

import_submodules(__name__, __path__)
