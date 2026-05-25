"""Generate story-writing prompts for the 29M anti-repetition RL stage.

Writes data/story_prompts/train.jsonl with rows shaped for the `rl`
dataset — {"question", "answer"} — where `question` is a short
story-writing instruction in the TinyStories domain the 29M model was
pretrained on (and that fits the alpaca instruction-following format it
was SFT'd into), and `answer` is an unused placeholder. The
`distinct_ngram` reward is reference-free: it scores only the model's own
completion, so there is no target answer to match.

The point of this stage is not a reference to hit but an *objective* to
optimize: sample several stories per prompt, reward the ones with more
lexical variety, and push the policy off the repetition loops a 29M model
falls into. See configs/29M/rl_stories.yaml.

Usage:
    python scripts/data/make_story_prompts.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

OUT_DIR = Path("data/story_prompts")

# Simple, child-level subjects — TinyStories vocabulary, so the 29M model
# has actually seen these words and can write fluently about them.
_SUBJECTS = [
    "a lost puppy",
    "a brave little mouse",
    "two best friends",
    "a magic tree",
    "a rainy day at the park",
    "a kitten who could not sleep",
    "a boy who found a shiny stone",
    "a girl and her red balloon",
    "a friendly dragon",
    "a snowman who wanted to fly",
    "a duck who lost its way home",
    "a tiny robot in the garden",
    "a bear looking for honey",
    "a star that fell from the sky",
    "an old toy left in the attic",
    "a rabbit who loved to paint",
    "a turtle racing a hare",
    "a little boat on a big sea",
    "a fox and a bunch of grapes",
    "a kind giant and a small town",
]

# Instruction templates — phrased the way alpaca-style SFT data is, so the
# prompts land inside the policy's instruction-following distribution.
_TEMPLATES = [
    "Write a short story about {subject}.",
    "Tell me a little story about {subject}.",
    "Write a simple bedtime story about {subject}.",
    "Make up a short story about {subject}.",
    "Write a few sentences telling a story about {subject}.",
]


def _examples(rng: random.Random) -> list[dict]:
    # Every (template, subject) pair → varied, deterministic prompt set.
    rows = [
        {"question": tmpl.format(subject=subj), "answer": ""}
        for subj in _SUBJECTS
        for tmpl in _TEMPLATES
    ]
    rng.shuffle(rows)
    return rows


def main() -> None:
    rng = random.Random(0)
    rows = _examples(rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "train.jsonl"
    # Write to a sibling tmp file then atomically rename — matches the
    # convention in make_rl_smoketest_data.py / download_rl_data.py so a
    # Ctrl-C mid-write can't leave the loader staring at a truncated file.
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    print(f"wrote {len(rows):>4d} rows to {path}")


if __name__ == "__main__":
    main()
