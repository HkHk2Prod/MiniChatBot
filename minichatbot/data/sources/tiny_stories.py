"""TinyStories source (Eldan & Li, 2023).

~470M tokens of synthetic children's stories with a deliberately
limited vocabulary (~3000 words). Designed so that sub-100M-parameter
language models can produce coherent narratives. Chinchilla-optimal
for ~25M-param models.

Reference: https://arxiv.org/abs/2305.07759
"""

from __future__ import annotations

from minichatbot.data.sources import SOURCE_REGISTRY
from minichatbot.data.sources.hf_dataset import HFDatasetSource


@SOURCE_REGISTRY.register("tiny_stories")
class TinyStoriesSource(HFDatasetSource):
    def __init__(
        self,
        split: str = "train",
        max_docs: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            dataset_name="roneneldan/TinyStories",
            split=split,
            text_field="text",
            max_docs=max_docs,
            cache_dir=cache_dir,
        )
