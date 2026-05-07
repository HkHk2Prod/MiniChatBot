from minichatbot.data.base import BaseDataset
from minichatbot.utils.registry import Registry, import_submodules

DATASET_REGISTRY: Registry[BaseDataset] = Registry("dataset")

# Skip 'base' (abstract) and 'collators'/'sources' (subpackages with their
# own registries that we don't want to fire from the dataset registry path).
import_submodules(__name__, __path__, skip=("base", "collators", "sources"))
