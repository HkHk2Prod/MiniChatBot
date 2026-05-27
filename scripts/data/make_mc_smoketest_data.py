"""Generate a tiny synthetic multiple-choice dataset for the DPO smoketest.

Writes `data/mc_smoketest/{train,val}.jsonl` with `{context, choices, gold}`
rows in the same shape the `mc` dataset / DPO stage consume. The task is
trivially learnable (pick the sensible continuation) so a short DPO run
should show multiple-choice accuracy and the gold-vs-distractor margin
climbing — a real signal that the stage trains, not just that it runs.

    python scripts/data/make_mc_smoketest_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

# (context, choices, gold_index). Gold is the coherent continuation; the
# distractors are plausible-looking but wrong, so raw next-token likelihood
# alone shouldn't trivially separate them without learning.
TRAIN: list[tuple[str, list[str], int]] = [
    ("The opposite of hot is", [" cold", " tall", " round"], 0),
    ("The opposite of up is", [" down", " green", " loud"], 0),
    ("The opposite of big is", [" small", " fast", " wet"], 0),
    ("The opposite of happy is", [" sad", " square", " early"], 0),
    ("The opposite of light is", [" dark", " sweet", " seven"], 0),
    ("A dog is a kind of", [" animal", " mountain", " number"], 0),
    ("A rose is a kind of", [" flower", " engine", " language"], 0),
    ("An apple is a kind of", [" fruit", " planet", " feeling"], 0),
    ("Two plus two equals", [" four", " purple", " Monday"], 0),
    ("The sun rises in the", [" east", " kitchen", " pocket"], 0),
    ("Water is made of hydrogen and", [" oxygen", " thunder", " velvet"], 0),
    ("A king lives in a", [" castle", " sandwich", " whisper"], 0),
    ("Birds can", [" fly", " photosynthesize", " multiply"], 0),
    ("Fish live in the", [" water", " desert", " library"], 0),
    ("The capital of France is", [" Paris", " orange", " Tuesday"], 0),
    ("Ice is frozen", [" water", " music", " courage"], 0),
    ("A baby cat is called a", [" kitten", " boulder", " sonnet"], 0),
    ("Bees make", [" honey", " gravity", " sentences"], 0),
    ("The grass is usually", [" green", " angry", " rectangular"], 0),
    ("You hear with your", [" ears", " elbows", " opinions"], 0),
]

VAL: list[tuple[str, list[str], int]] = [
    ("The opposite of fast is", [" slow", " blue", " heavy"], 0),
    ("The opposite of open is", [" closed", " bright", " three"], 0),
    ("A cow is a kind of", [" animal", " river", " idea"], 0),
    ("Three plus one equals", [" four", " silent", " Friday"], 0),
    ("The moon orbits the", [" earth", " spoon", " adjective"], 0),
    ("Snow is cold and", [" white", " hungry", " circular"], 0),
    ("A young dog is called a", [" puppy", " canyon", " theorem"], 0),
    ("You see with your", [" eyes", " ankles", " regrets"], 0),
]


def _write(path: Path, rows: list[tuple[str, list[str], int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for context, choices, gold in rows:
            f.write(json.dumps({"context": context, "choices": choices, "gold": gold}) + "\n")
    print(f"wrote {len(rows)} examples -> {path}")


def main() -> None:
    out = Path("data/mc_smoketest")
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "train.jsonl", TRAIN)
    _write(out / "val.jsonl", VAL)


if __name__ == "__main__":
    main()
