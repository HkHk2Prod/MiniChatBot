"""Train a BPE tokenizer from a text corpus.

Examples:
    python scripts/train_tokenizer.py --corpus data/corpus.txt --output tokenizer.json --vocab-size 32000
    python scripts/train_tokenizer.py --corpus data/corpus.jsonl --jsonl-key text --output tokenizer.json
    python scripts/train_tokenizer.py --corpus data/raw/ --output tokenizer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from minichatbot.tokenizer.bpe import BPETokenizer, IM_END_TOKEN, IM_START_TOKEN


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


def build_corpus_iterator(args: argparse.Namespace) -> Iterator[str]:
    path = Path(args.corpus)
    if path.is_dir():
        return _iter_directory(path)
    if args.jsonl_key:
        return _iter_jsonl(path, args.jsonl_key)
    return _iter_lines(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--corpus", required=True, help="Text file, JSONL file, or directory.")
    parser.add_argument("--output", required=True, help="Path to write tokenizer.json.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--jsonl-key",
        default=None,
        help="If corpus is a JSONL file, the key holding text content (e.g., 'text').",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide the BpeTrainer progress bar (useful when piping output to a log).",
    )
    args = parser.parse_args()

    corpus = build_corpus_iterator(args)
    print(
        f"Training BPE tokenizer "
        f"(vocab_size={args.vocab_size}, min_frequency={args.min_frequency}) ..."
    )
    tok = BPETokenizer.train(
        corpus=corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=not args.quiet,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)
    print(f"Saved tokenizer to {out}")
    print(f"  vocab_size  = {tok.vocab_size}")
    print(f"  pad_id      = {tok.pad_id}")
    print(f"  bos_id      = {tok.bos_id}")
    print(f"  eos_id      = {tok.eos_id}")
    print(f"  im_start_id = {tok.special_token_id(IM_START_TOKEN)}")
    print(f"  im_end_id   = {tok.special_token_id(IM_END_TOKEN)}")


if __name__ == "__main__":
    main()
