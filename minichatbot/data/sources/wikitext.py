"""WikiText-103 source for the wikitext DAPT branch.

Raw WikiText-103 train text — the in-domain corpus for lowering WikiText
test perplexity via domain-adaptive pretraining. Tokenize with the base
model's tokenizer through the normal download_corpus -> prepare_data path.
"""

from __future__ import annotations

from minichatbot.data.sources import SOURCE_REGISTRY
from minichatbot.data.sources.hf_dataset import HFDatasetSource


@SOURCE_REGISTRY.register("wikitext")
class WikiText103Source(HFDatasetSource):
    def __init__(
        self,
        split: str = "train",
        max_docs: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            dataset_name="Salesforce/wikitext",
            config_name="wikitext-103-raw-v1",
            text_field="text",
            split=split,
            max_docs=max_docs,
            cache_dir=cache_dir,
        )
