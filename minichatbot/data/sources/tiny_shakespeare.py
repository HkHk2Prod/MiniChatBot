"""Tiny Shakespeare source.

A single ~1MB text file from karpathy/char-rnn — the works of
Shakespeare concatenated. Useful for end-to-end pipeline sanity tests:
trains a small model in minutes, no `datasets` dependency required
(just urllib).

Each paragraph (separated by blank lines) is yielded as a separate
document so the tokenizer / packer treats them as independent EOS-
terminated sequences during pretraining.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from urllib.request import urlopen

from minichatbot.data.sources import SOURCE_REGISTRY
from minichatbot.data.sources.base import CorpusSource

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


@SOURCE_REGISTRY.register("tiny_shakespeare")
class TinyShakespeareSource(CorpusSource):
    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = cache_dir

    def iter_documents(self) -> Iterator[str]:
        path = self._download()
        text = path.read_text(encoding="utf-8")
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                yield para

    def _download(self) -> Path:
        if self.cache_dir is not None:
            cache = Path(self.cache_dir)
        else:
            cache = Path.home() / ".cache" / "minichatbot" / "tiny_shakespeare"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "tiny_shakespeare.txt"
        if not path.exists():
            with urlopen(URL) as resp:
                path.write_bytes(resp.read())
        return path

    def __repr__(self) -> str:
        return f"{type(self).__name__}(cache_dir={self.cache_dir!r})"
