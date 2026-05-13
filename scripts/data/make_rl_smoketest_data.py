"""Generate a tiny synthetic RL prompt set for pipeline smoketests.

Writes data/rl_smoketest/train.jsonl with GSM8K-shaped rows — a short
word problem as `question`, and an `answer` ending in the canonical
`#### <number>` marker that `GSM8KReward` parses. The point is not for
the model to actually solve these (a 2048-vocab toy model won't) — it's
to exercise the RL path end to end: prompt rendering → sampling a group
of completions → reward → advantage → policy-gradient step.

Usage:
    python scripts/data/make_rl_smoketest_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path("data/rl_smoketest")


def _examples(n: int, rng: random.Random) -> list[dict]:
    out = []
    for _ in range(n):
        a = rng.randint(2, 20)
        b = rng.randint(2, 20)
        kind = rng.choice(["add", "mul", "double"])
        if kind == "add":
            q = f"Tom has {a} apples and buys {b} more. How many apples does he have now?"
            steps = f"Tom starts with {a} and gets {b} more, so {a} + {b} = {a + b}."
            ans = a + b
        elif kind == "mul":
            q = f"A box holds {a} pencils. How many pencils are in {b} boxes?"
            steps = f"Each box has {a} pencils and there are {b} boxes, so {a} x {b} = {a * b}."
            ans = a * b
        else:
            q = f"Sara had {a} stickers and then doubled her collection. How many does she have?"
            steps = f"Doubling {a} gives {a} x 2 = {a * 2}."
            ans = a * 2
        out.append({"question": q, "answer": f"{steps}\n#### {ans}"})
    return out


def main() -> None:
    rng = random.Random(0)
    rows = _examples(200, rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>4d} rows to {path}")


if __name__ == "__main__":
    main()
