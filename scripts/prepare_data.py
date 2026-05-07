"""Tokenize a text/JSONL corpus into a packed uint16 binary for pretraining.

Examples:
    python scripts/prepare_data.py \\
        --corpus data/raw/ \\
        --tokenizer tokenizer.json \\
        --output data/

    python scripts/prepare_data.py \\
        --corpus data/big.jsonl \\
        --jsonl-key text \\
        --tokenizer tokenizer.json \\
        --output data/ \\
        --val-frac 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from minichatbot.tokenizer.bpe import BPETokenizer

UINT16_MAX = 65535
DOC_BATCH_SIZE = 1024


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


def _build_iterator(args: argparse.Namespace) -> Iterator[str]:
    path = Path(args.corpus)
    if path.is_dir():
        return _iter_directory(path)
    if args.jsonl_key:
        return _iter_jsonl(path, args.jsonl_key)
    return _iter_lines(path)


def tokenize_corpus(corpus: Iterator[str], tokenizer: BPETokenizer) -> np.ndarray:
    chunks: list[np.ndarray] = []
    batch: list[str] = []

    def flush() -> None:
        ids_batch = tokenizer.encode_batch(batch, add_special=True)
        for ids in ids_batch:
            chunks.append(np.asarray(ids, dtype=np.uint32))
        batch.clear()

    for text in tqdm(corpus, desc="tokenizing", unit=" docs"):
        batch.append(text)
        if len(batch) >= DOC_BATCH_SIZE:
            flush()
    if batch:
        flush()

    if not chunks:
        raise ValueError("Corpus produced no tokens.")
    arr = np.concatenate(chunks)
    if int(arr.max()) > UINT16_MAX:
        raise ValueError(
            f"Token id {int(arr.max())} exceeds uint16 max ({UINT16_MAX}). "
            f"Use a tokenizer with vocab_size <= 65536, or migrate the on-disk "
            f"format to uint32."
        )
    return arr.astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize a corpus into a packed uint16 .bin file."
    )
    parser.add_argument("--corpus", required=True, help="Text file, JSONL file, or directory of .txt.")
    parser.add_argument("--tokenizer", required=True, help="Path to a trained tokenizer.json.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory; writes train.bin (and val.bin if --val-frac > 0).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="Fraction of total tokens reserved at the end for validation.",
    )
    parser.add_argument(
        "--jsonl-key",
        default=None,
        help="If --corpus is a JSONL file, the field holding text content (e.g. 'text').",
    )
    args = parser.parse_args()

    if not 0.0 <= args.val_frac < 1.0:
        raise ValueError("--val-frac must be in [0, 1)")

    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"  vocab_size = {tokenizer.vocab_size}")

    corpus = _build_iterator(args)
    tokens = tokenize_corpus(corpus, tokenizer)
    n = len(tokens)
    print(f"Tokenized {n:,} tokens.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.val_frac > 0:
        val_n = int(n * args.val_frac)
        train_n = n - val_n
        train_path = out_dir / "train.bin"
        val_path = out_dir / "val.bin"
        tokens[:train_n].tofile(train_path)
        tokens[train_n:].tofile(val_path)
        print(f"  -> {train_path} ({train_n:,} tokens)")
        print(f"  -> {val_path}   ({val_n:,} tokens)")
    else:
        train_path = out_dir / "train.bin"
        tokens.tofile(train_path)
        print(f"  -> {train_path} ({n:,} tokens)")


if __name__ == "__main__":
    main()
