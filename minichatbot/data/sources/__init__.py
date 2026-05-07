from minichatbot.data.sources.base import CorpusSource
from minichatbot.utils.registry import Registry, import_submodules

SOURCE_REGISTRY: Registry[CorpusSource] = Registry("corpus_source")

import_submodules(__name__, __path__)
