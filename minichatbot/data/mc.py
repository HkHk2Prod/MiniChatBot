"""Multiple-choice dataset for the DPO stage: JSONL of
`{"context", "choices", "gold"}` rows.

Each line is one task instance — a shared context, a list of candidate
continuations, and the index of the correct (gold) one:

    {"context": "The opposite of hot is", "choices": [" cold", " loud"], "gold": 0}

Every (context, choice) pair is tokenized the *same way* the lm-eval
adapter scores multiple-choice tasks: encode `context + continuation`
jointly and slice off the context's tokens (BPE may merge across the
boundary, so this differs from encoding the two separately), after moving
any trailing context whitespace onto the continuation. That keeps the
training pairs identical to what the harness evaluates — so the DPO loss
optimizes exactly the quantity the benchmark ranks on. No chat template:
these are raw-text continuations, just like the eval.

`__getitem__` returns one example's candidate rows (each `context +
choice`, left-truncated to `seq_len`), the number of scored continuation
tokens and the continuation character length per candidate (for
length-normalized scoring), and the gold index. The `mc` collator
flattens a batch of these into model-ready rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from minichatbot.config import DataConfig
from minichatbot.data import DATASET_REGISTRY
from minichatbot.data.base import BaseDataset
from minichatbot.tokenizer.base import Tokenizer


def encode_continuation(
    tokenizer: Tokenizer, context: str, continuation: str, *, seq_len: int, prefix_id: int
) -> tuple[list[int], int, int]:
    """Tokenize one (context, continuation) pair the lm-eval way.

    Returns `(input_ids, n_cont, cont_chars)` where `input_ids` is
    `context + continuation` left-truncated to `seq_len` (oldest context
    dropped first; continuation always kept), `n_cont` is the number of
    continuation tokens actually scored (>=1 needs a preceding token to
    predict from), and `cont_chars` is the continuation's character length
    for length-normalized scoring.
    """
    # Trailing context whitespace is moved onto the continuation so the
    # token split lands on a boundary (lm-eval convention).
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    cont_chars = len(continuation)
    whole = tokenizer.encode(context + continuation, include_special=False)
    ctx_enc = tokenizer.encode(context, include_special=False) if context else []
    n_cont_full = len(whole) - len(ctx_enc)

    inp = whole if ctx_enc else [prefix_id] + whole
    if not ctx_enc:
        # Empty context: a leading prefix token supplies the conditioning
        # the first continuation token is predicted from.
        n_cont_full = len(whole)
    if len(inp) > seq_len:
        inp = inp[-seq_len:]
    n_cont = max(min(n_cont_full, len(inp) - 1), 0)
    return inp, n_cont, cont_chars


@DATASET_REGISTRY.register("mc")
class MCDataset(BaseDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.prefix_id = tokenizer.bos_id if tokenizer.bos_id is not None else tokenizer.eos_id
        # Each example: (rows, n_conts, cont_chars, gold)
        self.examples: list[tuple[list[list[int]], list[int], list[int], int]] = []
        self._load()

    def _load(self) -> None:
        n_total = 0
        n_dropped = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                obj = json.loads(line)
                context = str(obj["context"])
                choices = [str(c) for c in obj["choices"]]
                gold = int(obj["gold"])
                if len(choices) < 2 or not (0 <= gold < len(choices)):
                    n_dropped += 1
                    continue
                rows: list[list[int]] = []
                n_conts: list[int] = []
                cont_chars: list[int] = []
                ok = True
                for choice in choices:
                    inp, n_cont, chars = encode_continuation(
                        self.tokenizer, context, choice,
                        seq_len=self.seq_len, prefix_id=self.prefix_id,
                    )
                    if n_cont == 0:
                        ok = False
                        break
                    rows.append(inp)
                    n_conts.append(n_cont)
                    cont_chars.append(chars)
                if not ok:
                    n_dropped += 1
                    continue
                self.examples.append((rows, n_conts, cont_chars, gold))
        if n_dropped:
            print(
                f"[mc-dataset] dropped {n_dropped}/{n_total} examples "
                f"(fewer than 2 choices, bad gold index, or empty continuation)."
            )
        if not self.examples:
            raise ValueError(
                f"MCDataset: no usable examples in {self.path} "
                f"(checked {n_total} lines)."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        rows, n_conts, cont_chars, gold = self.examples[idx]
        return {
            "input_rows": [torch.tensor(r, dtype=torch.long) for r in rows],
            "n_conts": torch.tensor(n_conts, dtype=torch.long),
            "cont_chars": torch.tensor(cont_chars, dtype=torch.long),
            "gold": gold,
        }

    @classmethod
    def from_config(
        cls,
        cfg: DataConfig,
        tokenizer: Tokenizer,
        split: str = "train",
    ) -> MCDataset:
        if split == "train":
            path = cfg.train_path
        elif split == "val":
            if cfg.val_path is None:
                raise ValueError(
                    "MCDataset.from_config(split='val') requires "
                    "DataConfig.val_path to be set."
                )
            path = cfg.val_path
        else:
            raise ValueError(f"Unknown split: {split!r}")
        return cls(path=path, tokenizer=tokenizer, seq_len=cfg.seq_len)
