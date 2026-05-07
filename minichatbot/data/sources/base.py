"""Base class for corpus sources.

A `CorpusSource` is anything that yields text documents — HF datasets,
URLs, local files, etc. Sources normalize disparate transports to a
single shape (an iterator of strings, one per logical document) so the
downstream pipeline (tokenize, pack into .bin) doesn't care about
where the text came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class CorpusSource(ABC):
    """Yields text documents from some source."""

    @abstractmethod
    def iter_documents(self) -> Iterator[str]: ...

    def __iter__(self) -> Iterator[str]:
        return self.iter_documents()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
