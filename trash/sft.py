"""Deprecated alias for `scripts/train/train.py`, with the stage forced to
"sft". The stage now lives in the config (`stage:` / `data.type`), so prefer
`train.py` directly. Kept so existing tasks/commands keep working.
"""

from __future__ import annotations

import sys

from minichatbot.training.cli import train_main

if __name__ == "__main__":
    train_main(sys.argv[1:] + ["--stage", "sft"])
