"""Generate a tiny synthetic SFT dataset for pipeline smoketests.

Writes data/sft_smoketest/{train,val}.jsonl with simple Q&A pairs:
arithmetic, greetings, and one-shot echoes. The point is not to teach
the model anything useful — it's to exercise the SFT path (chat
template → masked labels → trainer loop → samples) on real-looking
data without needing to download anything.

Usage:
    python scripts/make_sft_smoketest_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path("data/sft_smoketest")


def _arithmetic_examples(n: int, rng: random.Random) -> list[dict]:
    out = []
    for _ in range(n):
        op = rng.choice(["+", "-", "*"])
        a = rng.randint(1, 12)
        b = rng.randint(1, 12)
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b
        question = rng.choice(
            [f"What is {a} {op} {b}?", f"Compute {a} {op} {b}.", f"{a} {op} {b} = ?"]
        )
        out.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"{ans}."},
            ]
        })
    return out


def _greeting_examples(n: int, rng: random.Random) -> list[dict]:
    pairs = [
        ("Hi", "Hello!"),
        ("Hello", "Hi there!"),
        ("Hey", "Hey, how are you?"),
        ("Good morning", "Good morning!"),
        ("How are you?", "I'm doing well, thanks."),
        ("What's up?", "Not much, just here to help."),
    ]
    out = []
    for _ in range(n):
        q, a = rng.choice(pairs)
        out.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })
    return out


def _echo_examples(n: int, rng: random.Random) -> list[dict]:
    words = ["cat", "dog", "tree", "river", "moon", "stone", "bird", "fire", "wind", "leaf"]
    out = []
    for _ in range(n):
        w = rng.choice(words)
        out.append({
            "messages": [
                {"role": "user", "content": f"Repeat: {w}"},
                {"role": "assistant", "content": w},
            ]
        })
    return out


def main() -> None:
    rng = random.Random(0)
    train = (
        _arithmetic_examples(150, rng)
        + _greeting_examples(60, rng)
        + _echo_examples(60, rng)
    )
    val = (
        _arithmetic_examples(20, rng)
        + _greeting_examples(10, rng)
        + _echo_examples(10, rng)
    )
    rng.shuffle(train)
    rng.shuffle(val)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        path = OUT_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>4d} rows to {path}")


if __name__ == "__main__":
    main()
