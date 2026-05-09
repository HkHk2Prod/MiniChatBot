"""Download a registered corpus source and write it to a JSONL file.

Output format: one JSON object per line, with a `text` field.
This is consumable by scripts/data/prepare_data.py via `--jsonl-key text`.

Examples:
    # Sanity check (~1MB)
    python scripts/data/download_corpus.py --source tiny_shakespeare --output data/shakespeare.jsonl

    # First real run (~2GB; --max-docs caps it for testing)
    python scripts/data/download_corpus.py --source tiny_stories --output data/tinystories.jsonl
    python scripts/data/download_corpus.py --source tiny_stories --output data/tinystories_small.jsonl --max-docs 100000

    # Big-model territory (TBs)
    python scripts/data/download_corpus.py --source fineweb_edu --output data/fineweb.jsonl \\
        --subset sample-10BT --max-docs 1000000

Pipeline (3 steps, each its own script):
    1. download_corpus.py    source key  -> data/corpus.jsonl
    2. train_tokenizer.py    corpus     -> tokenizer.json
    3. prepare_data.py       corpus + tokenizer -> data/{train,val}.bin
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from tqdm import tqdm

from minichatbot.data.sources import SOURCE_REGISTRY


def _build_source(args: argparse.Namespace):
    cls = SOURCE_REGISTRY[args.source]
    candidate_kwargs: dict[str, object] = {
        "split": args.split,
        "max_docs": args.max_docs,
        "cache_dir": args.cache_dir,
        "subset": args.subset,
    }
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted and v is not None}

    if args.subset is not None and "subset" not in accepted:
        print(f"warning: --subset is ignored for source {args.source!r}")

    return cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a corpus source to a JSONL file.")
    parser.add_argument(
        "--source",
        required=True,
        help=f"Source registry key. Available: {sorted(SOURCE_REGISTRY.keys())}",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap on documents to fetch. Useful for sanity tests.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Subset / config (e.g., 'sample-10BT' for fineweb_edu).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for the underlying source.",
    )
    args = parser.parse_args()

    source = _build_source(args)
    print(f"Source: {source!r}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    n_chars = 0
    with out.open("w", encoding="utf-8") as f:
        for doc in tqdm(source, desc="downloading", unit=" docs"):
            if not doc:
                continue
            f.write(json.dumps({"text": doc}, ensure_ascii=False))
            f.write("\n")
            n_docs += 1
            n_chars += len(doc)
    print(f"Wrote {n_docs:,} docs ({n_chars:,} chars) to {out}")
    print(
        f"\nNext steps:\n"
        f"  python scripts/data/train_tokenizer.py --corpus {out} --jsonl-key text "
        f"--output tokenizer.json --vocab-size 8000\n"
        f"  python scripts/data/prepare_data.py --corpus {out} --jsonl-key text "
        f"--tokenizer tokenizer.json --output data/ --val-frac 0.005"
    )


if __name__ == "__main__":
    main()
