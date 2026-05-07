"""FineWeb-Edu source (HuggingFaceFW).

High-quality educational web text, filtered from Common Crawl with an
edu-classifier. Standard pretraining corpus circa 2024-2025; better
signal/token than older WebText/OpenWebText.

Subsets:
    sample-10BT    ~26GB, 10B tokens   (recommended for ~100-300M models)
    sample-100BT   ~260GB, 100B tokens
    sample-350BT   ~900GB, 350B tokens
    default        full corpus

Reference: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

from __future__ import annotations

from minichatbot.data.sources import SOURCE_REGISTRY
from minichatbot.data.sources.hf_dataset import HFDatasetSource

VALID_SUBSETS = {"sample-10BT", "sample-100BT", "sample-350BT", "default"}


@SOURCE_REGISTRY.register("fineweb_edu")
class FineWebEduSource(HFDatasetSource):
    def __init__(
        self,
        subset: str = "sample-10BT",
        split: str = "train",
        max_docs: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if subset not in VALID_SUBSETS:
            raise ValueError(
                f"FineWeb-Edu subset must be one of {sorted(VALID_SUBSETS)}, got {subset!r}"
            )
        super().__init__(
            dataset_name="HuggingFaceFW/fineweb-edu",
            config_name=subset,
            split=split,
            text_field="text",
            max_docs=max_docs,
            cache_dir=cache_dir,
        )
