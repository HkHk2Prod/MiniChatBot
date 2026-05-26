"""Tokenize a text/JSONL corpus into a packed uint16 binary for pretraining.

Examples:
    python scripts/data/prepare_data.py \\
        --corpus data/raw/ \\
        --tokenizer tokenizer.json \\
        --output data/

    python scripts/data/prepare_data.py \\
        --corpus data/big.jsonl \\
        --jsonl-key text \\
        --tokenizer tokenizer.json \\
        --output data/ \\
        --val-frac 0.001
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from minichatbot.data.corpus_iter import build_corpus_iterator
from minichatbot.tokenizer.bpe import BPETokenizer

UINT16_MAX = 65535
DOC_BATCH_SIZE = 1024
ITEMSIZE = np.dtype(np.uint16).itemsize  # bytes per token on disk
COPY_CHUNK_BYTES = 8 * 1024 * 1024


def tokenize_to_bin(corpus: Iterator[str], tokenizer: BPETokenizer, out_path: Path) -> int:
    """Stream-tokenize `corpus`, appending uint16 token ids to `out_path`.

    Memory stays bounded by one ``DOC_BATCH_SIZE`` batch instead of the whole
    corpus: each batch is encoded, written, and discarded. Returns the total
    number of tokens written.
    """
    total = 0
    batch: list[str] = []

    with open(out_path, "wb") as fh:

        def flush() -> None:
            nonlocal total
            if not batch:
                return
            ids_batch = tokenizer.encode_batch(batch, include_special=True)
            batch.clear()
            arr = np.concatenate([np.asarray(ids, dtype=np.uint32) for ids in ids_batch])
            if arr.size and int(arr.max()) > UINT16_MAX:
                raise ValueError(
                    f"Token id {int(arr.max())} exceeds uint16 max ({UINT16_MAX}). "
                    f"Use a tokenizer with vocab_size <= 65536, or migrate the on-disk "
                    f"format to uint32."
                )
            arr.astype(np.uint16).tofile(fh)
            total += int(arr.size)

        for text in tqdm(corpus, desc="tokenizing", unit=" docs"):
            batch.append(text)
            if len(batch) >= DOC_BATCH_SIZE:
                flush()
        flush()

    if total == 0:
        out_path.unlink(missing_ok=True)
        raise ValueError("Corpus produced no tokens.")
    return total


def split_tail(train_path: Path, val_path: Path, train_n: int, val_n: int) -> None:
    """Move the last `val_n` tokens of `train_path` into `val_path` in place.

    Copies the tail in fixed-size chunks then truncates the train file, so peak
    memory is one ``COPY_CHUNK_BYTES`` buffer rather than the validation split.
    """
    train_bytes = train_n * ITEMSIZE
    with open(train_path, "rb") as src, open(val_path, "wb") as dst:
        src.seek(train_bytes)
        remaining = val_n * ITEMSIZE
        while remaining > 0:
            chunk = src.read(min(remaining, COPY_CHUNK_BYTES))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    with open(train_path, "r+b") as fh:
        fh.truncate(train_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize a corpus into a packed uint16 .bin file."
    )
    parser.add_argument(
        "--corpus", required=True, help="Text file, JSONL file, or directory of .txt."
    )
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

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"

    corpus = build_corpus_iterator(args.corpus, jsonl_key=args.jsonl_key)
    n = tokenize_to_bin(corpus, tokenizer, train_path)
    print(f"Tokenized {n:,} tokens.")

    if args.val_frac > 0:
        val_n = int(n * args.val_frac)
        train_n = n - val_n
        val_path = out_dir / "val.bin"
        split_tail(train_path, val_path, train_n, val_n)
        print(f"  -> {train_path} ({train_n:,} tokens)")
        print(f"  -> {val_path}   ({val_n:,} tokens)")
    else:
        print(f"  -> {train_path} ({n:,} tokens)")


if __name__ == "__main__":
    main()
