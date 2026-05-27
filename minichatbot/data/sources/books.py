"""BookCorpusOpen source for the lambada DAPT branch.

Long-form narrative prose — the in-domain corpus for LAMBADA, whose
last-word-prediction passages are drawn from books. Stream-capped via
--max-docs (full BookCorpusOpen is large); feed through the normal
download_corpus -> prepare_data path with the base model's tokenizer.
"""

from __future__ import annotations

from minichatbot.data.sources import SOURCE_REGISTRY
from minichatbot.data.sources.hf_dataset import HFDatasetSource


@SOURCE_REGISTRY.register("books")
class BooksSource(HFDatasetSource):
    def __init__(
        self,
        split: str = "train",
        max_docs: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            dataset_name="lucadiliello/bookcorpusopen",
            text_field="text",
            split=split,
            max_docs=max_docs,
            cache_dir=cache_dir,
        )
