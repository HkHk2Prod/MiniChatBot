"""Tests for the corpus iterator (minichatbot/data/corpus_iter.py)."""

from __future__ import annotations

import json
from pathlib import Path

from minichatbot.data.corpus_iter import build_corpus_iterator


def test_iterates_txt_lines_skipping_blanks(tmp_path: Path) -> None:
    f = tmp_path / "corpus.txt"
    f.write_text("first line\n\nsecond line\n", encoding="utf-8")
    assert list(build_corpus_iterator(f)) == ["first line", "second line"]


def test_iterates_jsonl_by_key(tmp_path: Path) -> None:
    f = tmp_path / "corpus.jsonl"
    rows = [{"text": "hello"}, {"text": ""}, {"other": "ignored"}, {"text": "world"}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    # Empty/absent values for the key are skipped.
    assert list(build_corpus_iterator(f, jsonl_key="text")) == ["hello", "world"]


def test_walks_directory_of_txt_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "skip.md").write_text("not txt\n", encoding="utf-8")
    # rglob('*.txt') is sorted -> a before b; non-.txt ignored.
    assert list(build_corpus_iterator(tmp_path)) == ["alpha", "beta"]


def test_directory_takes_precedence_over_jsonl_key(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("plain\n", encoding="utf-8")
    # Passing a directory dispatches to the directory walk even with jsonl_key set.
    assert list(build_corpus_iterator(tmp_path, jsonl_key="text")) == ["plain"]
