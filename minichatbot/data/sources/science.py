"""SciQ-based science corpus for domain-adaptive pretraining (DAPT).

Streams the SciQ split and emits, per example, its support passage
(grade-school science prose) plus a rendered "Question: ...\\nAnswer: ..."
line from the question + correct answer. This is the ARC branch's DAPT
material: raw science text that raises the model's prior on correct
ARC-style continuations, in the same format the eval scores — fed through
the normal `download_corpus.py -> prepare_data.py` pipeline (tokenized with
the *base model's* tokenizer, not a new one).

    python scripts/data/download_corpus.py --source science --output data/science/corpus.jsonl
    python scripts/data/prepare_data.py --corpus data/science/corpus.jsonl --jsonl-key text \\
        --tokenizer data/fineweb/tokenizer.json --output data/science/ --val-frac 0.02
"""

from __future__ import annotations

from collections.abc import Iterator

from minichatbot.data.sources import SOURCE_REGISTRY
from minichatbot.data.sources.base import CorpusSource


@SOURCE_REGISTRY.register("science")
class ScienceSource(CorpusSource):
    def __init__(
        self,
        split: str = "train",
        max_docs: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.split = split
        self.max_docs = max_docs
        self.cache_dir = cache_dir

    def __iter__(self) -> Iterator[str]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "ScienceSource requires the `datasets` package. "
                'Install with: pip install -e ".[data]"'
            ) from e

        # Fully-qualified id: newer huggingface_hub rejects the bare canonical
        # name "sciq" ("Repository id must be 'namespace/name'").
        ds = load_dataset(
            "allenai/sciq", split=self.split, streaming=True, cache_dir=self.cache_dir
        )
        n = 0
        for row in ds:
            if self.max_docs is not None and n >= self.max_docs:
                break
            support = (row.get("support") or "").strip()
            question = (row.get("question") or "").strip()
            answer = (row.get("correct_answer") or "").strip()
            parts: list[str] = []
            if support:
                parts.append(support)
            if question and answer:
                parts.append(f"Question: {question}\nAnswer: {answer}")
            text = "\n\n".join(parts).strip()
            if text:
                yield text
                n += 1

    def __repr__(self) -> str:
        cap = f", max_docs={self.max_docs}" if self.max_docs is not None else ""
        return f"ScienceSource(dataset='allenai/sciq', split={self.split!r}{cap})"
