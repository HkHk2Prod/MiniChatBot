"""Download an RL prompt set from HuggingFace and write it as {question, answer} JSONL.

Output format (one task per line, consumable by RLPromptDataset):
    {"question": "...", "answer": "...  #### 42"}

The `answer` is whatever the reward function checks completions against —
for GSM8K it keeps the original chain-of-thought ending in the canonical
`#### <number>` marker, which `GSM8KReward` parses.

This is a sibling of `download_sft_data.py` (chat conversations) and
`download_corpus.py` (plain text): RL needs prompts + a reference answer,
not a target response. Different shape, different downstream pipeline.

Examples:
    # GSM8K (7.5K train + 1.3K test grade-school math problems)
    python scripts/data/download_rl_data.py --source gsm8k --output-dir data/gsm8k

    # Cap to a sanity-test size
    python scripts/data/download_rl_data.py --source gsm8k --output-dir data/gsm8k_small \\
        --max-rows 200

Available sources: see SOURCES below — each is a
(hf_name, hf_config, {split: hf_split}, row_to_qa_fn).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

# (question, answer) or None to skip the row.
RowToQA = Callable[[dict], "tuple[str, str] | None"]


def _gsm8k_to_qa(row: dict) -> "tuple[str, str] | None":
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not question or not answer or "####" not in answer:
        return None
    return question, answer


# name -> (hf_dataset, hf_config, {output_split: hf_split}, row_fn)
SOURCES: dict[str, tuple[str, str | None, dict[str, str], RowToQA]] = {
    "gsm8k": ("gsm8k", "main", {"train": "train", "test": "test"}, _gsm8k_to_qa),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an RL prompt set to {question,answer} JSONL.")
    parser.add_argument("--source", required=True, choices=sorted(SOURCES.keys()), help="RL source key.")
    parser.add_argument("--output-dir", required=True, help="Directory for the per-split .jsonl files.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap on rows per split. Useful for sanity tests.",
    )
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache directory.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "download_rl_data.py requires the `datasets` package. "
            'Install with: pip install -e ".[data]"   (or pip install "datasets>=2.20")'
        ) from e

    hf_name, hf_config, split_map, to_qa = SOURCES[args.source]
    print(f"Source: {args.source} ({hf_name}{'/' + hf_config if hf_config else ''})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for out_split, hf_split in split_map.items():
        ds = load_dataset(hf_name, hf_config, split=hf_split, cache_dir=args.cache_dir)
        path = out_dir / f"{out_split}.jsonl"
        n_written = 0
        n_skipped = 0
        with path.open("w", encoding="utf-8") as f:
            for i, row in enumerate(tqdm(ds, desc=out_split, unit=" rows")):
                if args.max_rows is not None and i >= args.max_rows:
                    break
                qa = to_qa(row)
                if qa is None:
                    n_skipped += 1
                    continue
                question, answer = qa
                f.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False))
                f.write("\n")
                n_written += 1
        msg = f"wrote {n_written:>6,} rows to {path}"
        if n_skipped:
            msg += f"  (skipped {n_skipped} malformed)"
        print(msg)

    print(
        f"\nNext step: point your RL config's data.train_path at\n"
        f"  {out_dir / 'train.jsonl'}\n"
        f"then run scripts/train/rl.py --config configs/100M/rl_gsm8k.yaml --from-pretrained auto"
    )


if __name__ == "__main__":
    main()
