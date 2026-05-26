"""Build multiple-choice train/val JSONL for the DPO stage from an lm-eval task.

Renders the task's train (and validation) docs with lm-eval's OWN
`doc_to_text` / `doc_to_choice` / `doc_to_target`, so the (context, choice)
pairs are format-identical to what the harness scores at eval time on the
held-out split — the DPO loss then optimizes exactly the quantity the
benchmark ranks. Each candidate string includes the task's target
delimiter (a leading space for ARC/PIQA/HellaSwag), matching how the
harness joins context and continuation.

Output rows (one per line): {"context": str, "choices": [str], "gold": int}.
Only train/val are emitted — the harness's own test/validation split stays
held out for evaluation, so training never sees the eval examples.

Requires the optional `eval` extra:
    pip install -e ".[eval]"

    python scripts/data/make_mc_data.py --task arc_easy --output-dir data/arc_easy
    python scripts/data/make_mc_data.py --task hellaswag --output-dir data/hellaswag --limit 20000
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any


def _gold_index(target: Any, choices: list[str]) -> int | None:
    """lm-eval `doc_to_target` is usually the gold index, occasionally the
    gold string. Normalize to an index into `choices`, or None if unusable."""
    if isinstance(target, bool):  # bool is an int subclass; reject explicitly
        return None
    if isinstance(target, int):
        return target
    if isinstance(target, str) and target in choices:
        return choices.index(target)
    try:
        return int(target)
    except (TypeError, ValueError):
        return None


def _render(task: Any, docs: list[dict[str, Any]], delimiter: str) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for doc in docs:
        context = task.doc_to_text(doc)
        choices = task.doc_to_choice(doc)
        target = task.doc_to_target(doc)
        if not isinstance(context, str) or not isinstance(choices, (list, tuple)) or len(choices) < 2:
            skipped += 1
            continue
        choices = [str(c) for c in choices]
        gold = _gold_index(target, choices)
        if gold is None or not (0 <= gold < len(choices)):
            skipped += 1
            continue
        # Continuation = target_delimiter + choice, matching how the harness
        # joins context and continuation before scoring.
        rows.append(
            {"context": context, "choices": [f"{delimiter}{c}" for c in choices], "gold": gold}
        )
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--task", required=True, help="lm-eval task name (e.g. arc_easy, piqa, hellaswag).")
    ap.add_argument("--output-dir", required=True, help="Directory for {train,val}.jsonl.")
    ap.add_argument("--limit", type=int, default=None, help="Cap docs per split.")
    args = ap.parse_args()

    try:
        from lm_eval.tasks import get_task_dict
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "lm-evaluation-harness is not installed. Install the optional extra:\n"
            '    pip install -e ".[eval]"'
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        task = get_task_dict([args.task])[args.task]

    delimiter = getattr(getattr(task, "config", None), "target_delimiter", " ")

    splits: list[tuple[str, Any]] = []
    if task.has_training_docs():
        splits.append(("train", task.training_docs()))
        if task.has_validation_docs():
            splits.append(("val", task.validation_docs()))
    elif task.has_validation_docs():
        # No train split: train on validation (the harness scores on test).
        splits.append(("train", task.validation_docs()))
    else:
        raise SystemExit(f"task {args.task!r} exposes neither train nor validation docs.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, doc_iter in splits:
        docs = list(doc_iter)
        if args.limit is not None:
            docs = docs[: args.limit]
        rows, skipped = _render(task, docs, delimiter)
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{args.task} [{name}]: wrote {len(rows)} rows (skipped {skipped}) -> {path}")


if __name__ == "__main__":
    main()
