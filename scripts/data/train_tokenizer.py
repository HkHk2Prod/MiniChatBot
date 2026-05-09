"""Train a BPE tokenizer from a text corpus.

Examples:
    python scripts/data/train_tokenizer.py --corpus data/corpus.txt --output tokenizer.json --vocab-size 32000
    python scripts/data/train_tokenizer.py --corpus data/corpus.jsonl --jsonl-key text --output tokenizer.json
    python scripts/data/train_tokenizer.py --corpus data/raw/ --output tokenizer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minichatbot.data.corpus_iter import build_corpus_iterator
from minichatbot.tokenizer.bpe import BPETokenizer, IM_END_TOKEN, IM_START_TOKEN


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

    corpus = build_corpus_iterator(args.corpus, jsonl_key=args.jsonl_key)
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
