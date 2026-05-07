"""Generic HuggingFace `datasets` source.

Streams documents from any HF dataset that exposes a text field.
Specific datasets (TinyStories, FineWeb-Edu, etc.) subclass this with
their dataset name and field hard-coded.

Uses streaming mode by default so we never load the full corpus into
RAM — important for FineWeb-Edu (TBs) and TinyStories (~2GB).
"""

from __future__ import annotations

from collections.abc import Iterator

from minichatbot.data.sources.base import CorpusSource


class HFDatasetSource(CorpusSource):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        config_name: str | None = None,
        max_docs: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.config_name = config_name
        self.max_docs = max_docs
        self.cache_dir = cache_dir

    def iter_documents(self) -> Iterator[str]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HFDatasetSource requires the `datasets` package. "
                'Install with: pip install -e ".[data]"   '
                "(or pip install \"datasets>=2.20\")"
            ) from e

        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=True,
            cache_dir=self.cache_dir,
        )
        for i, row in enumerate(ds):
            if self.max_docs is not None and i >= self.max_docs:
                break
            text = row.get(self.text_field)
            if text:
                yield text

    def __repr__(self) -> str:
        config = f", config={self.config_name!r}" if self.config_name else ""
        cap = f", max_docs={self.max_docs}" if self.max_docs is not None else ""
        return f"{type(self).__name__}(dataset={self.dataset_name!r}, split={self.split!r}{config}{cap})"
