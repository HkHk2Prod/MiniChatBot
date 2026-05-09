"""Raw text corpus iterators shared by tokenizer training and data prep.

Yields one document/line at a time from a `.txt` file, a `.jsonl` file, or a
directory of `.txt` files. Pure iteration — no tokenization, no batching.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def _iter_jsonl(path: Path, key: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(key)
            if text:
                yield text


def _iter_directory(root: Path) -> Iterator[str]:
    for p in sorted(root.rglob("*.txt")):
        yield from _iter_lines(p)


def build_corpus_iterator(corpus: str | Path, jsonl_key: str | None = None) -> Iterator[str]:
    """Dispatch on path shape: directory -> walk *.txt; jsonl_key set -> JSONL; else lines."""
    path = Path(corpus)
    if path.is_dir():
        return _iter_directory(path)
    if jsonl_key:
        return _iter_jsonl(path, jsonl_key)
    return _iter_lines(path)
