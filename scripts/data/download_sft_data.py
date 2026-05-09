"""Download an SFT dataset from HuggingFace and write it as chat-format JSONL.

Output format (one conversation per line, consumable by SFTDataset):
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

Why this is a separate script from `download_corpus.py`: corpus sources
emit plain text (one document per line, `{"text": "..."}`), SFT sources
emit structured conversations. Different output shape, different downstream
pipeline (corpus → tokenize+pack → .bin; SFT → load + tokenize at runtime).

Examples:
    # Alpaca-cleaned (~52K instruction/response pairs, ~25 MB)
    python scripts/data/download_sft_data.py --source alpaca_cleaned \\
        --output-dir data/alpaca

    # Cap to a sanity-test size
    python scripts/data/download_sft_data.py --source alpaca_cleaned \\
        --output-dir data/alpaca_small --max-rows 2000

Available sources: see SOURCES dict below. Adding more is one entry —
each source is a (hf_dataset_name, row_to_messages_fn) pair.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path

from tqdm import tqdm

# A row is whatever HF's `datasets.load_dataset` yields per record.
# A messages list is the chat-format conversation our SFTDataset expects.
RowToMessages = Callable[[dict], list[dict] | None]


def _alpaca_to_messages(row: dict) -> list[dict] | None:
    """yahma/alpaca-cleaned schema: {instruction, input, output}.

    `input` is optional context; when present we append it to the user
    turn separated by a blank line. Returns None for rows missing the
    instruction or output (a few are blank in the cleaned dump).
    """
    instruction = (row.get("instruction") or "").strip()
    user_input = (row.get("input") or "").strip()
    output = (row.get("output") or "").strip()
    if not instruction or not output:
        return None
    user_content = f"{instruction}\n\n{user_input}" if user_input else instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


# (hf_dataset_name, row_to_messages_fn). Add more SFT datasets here.
SOURCES: dict[str, tuple[str, RowToMessages]] = {
    "alpaca_cleaned": ("yahma/alpaca-cleaned", _alpaca_to_messages),
}


def _stream_rows(
    dataset_name: str,
    max_rows: int | None,
    cache_dir: str | None,
) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "download_sft_data.py requires the `datasets` package. "
            'Install with: pip install -e ".[data]"   '
            "(or pip install \"datasets>=2.20\")"
        ) from e
    # Streaming so we don't load the full dataset into RAM. Alpaca-cleaned
    # is small enough to fit comfortably (~25 MB) but the same script
    # should scale to bigger sources without changes.
    ds = load_dataset(dataset_name, split="train", streaming=True, cache_dir=cache_dir)
    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an SFT dataset to chat-format JSONL.")
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(SOURCES.keys()),
        help="SFT source key.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for {train,val}.jsonl.")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.02,
        help="Fraction held out for val (default: 0.02 — ~1K rows out of alpaca's 52K).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap on rows to fetch (before train/val split). Useful for sanity tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train/val split.",
    )
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache directory.")
    args = parser.parse_args()

    if not 0.0 < args.val_frac < 1.0:
        raise SystemExit(f"--val-frac must be in (0, 1), got {args.val_frac}")

    hf_name, to_messages = SOURCES[args.source]
    print(f"Source: {args.source} ({hf_name})")

    # Materialize all valid conversations first so we can do a deterministic
    # train/val split. Alpaca-cleaned at ~52K rows fits in RAM comfortably;
    # for genuinely large SFT corpora we'd switch to a streaming split.
    conversations: list[list[dict]] = []
    n_skipped = 0
    for row in tqdm(
        _stream_rows(hf_name, args.max_rows, args.cache_dir),
        desc="downloading",
        unit=" rows",
    ):
        messages = to_messages(row)
        if messages is None:
            n_skipped += 1
            continue
        conversations.append(messages)
    if n_skipped:
        print(f"skipped {n_skipped} rows missing instruction or output")
    if not conversations:
        raise SystemExit("No conversations produced — nothing to write.")

    rng = random.Random(args.seed)
    rng.shuffle(conversations)
    n_val = max(1, int(len(conversations) * args.val_frac))
    val = conversations[:n_val]
    train = conversations[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for messages in rows:
                f.write(json.dumps({"messages": messages}, ensure_ascii=False))
                f.write("\n")
        print(f"wrote {len(rows):>6,} rows to {path}")

    print(
        f"\nNext step: point your SFT config's data.train_path / data.val_path at\n"
        f"  {out_dir / 'train.jsonl'}\n"
        f"  {out_dir / 'val.jsonl'}\n"
        f"then run scripts/train/sft.py."
    )


if __name__ == "__main__":
    main()
