"""Prompt dataset for the RL stage: JSONL of {"question", "answer"} rows.

Each line is one task instance. Unlike the SFT dataset (which carries the
target response), an RL example carries only the *prompt* the policy will
be asked to complete plus a *reference* string the reward function checks
its completion against. For GSM8K-style data:

    {"question": "Natalia sold clips to ...", "answer": "She sold ... #### 72"}

The question is rendered as a single user turn (optionally behind a
system prompt from `data.system_prompt`) and tokenized up to and
including the `<|im_start|>assistant\\n` header, leaving the model
positioned to generate. Prompts longer than `seq_len` are dropped (with
a count) — truncating a math question would change its answer.

`__getitem__` returns `{"prompt_ids": LongTensor, "reference": str}`;
batch them with the `rl` collator (default-collate can't stack
variable-length prompts or strings).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from minichatbot.chat.template import render_prompt_for_completion
from minichatbot.config import DataConfig
from minichatbot.data import DATASET_REGISTRY
from minichatbot.data.base import BaseDataset
from minichatbot.tokenizer.base import Tokenizer


@DATASET_REGISTRY.register("rl")
class RLPromptDataset(BaseDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        system_prompt: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.system_prompt = system_prompt
        self.examples: list[tuple[list[int], str]] = []
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
                question = obj["question"]
                reference = str(obj["answer"])
                messages: list[dict[str, str]] = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": question})
                prompt_ids = render_prompt_for_completion(messages, self.tokenizer)
                if len(prompt_ids) > self.seq_len:
                    n_dropped += 1
                    continue
                self.examples.append((prompt_ids, reference))
        if n_dropped:
            print(
                f"[rl-dataset] dropped {n_dropped}/{n_total} prompts longer than "
                f"seq_len={self.seq_len}."
            )
        if not self.examples:
            raise ValueError(
                f"RLPromptDataset: no usable prompts in {self.path} "
                f"(all {n_total} were empty or too long for seq_len={self.seq_len})."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        prompt_ids, reference = self.examples[idx]
        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "reference": reference,
        }

    @classmethod
    def from_config(
        cls,
        cfg: DataConfig,
        tokenizer: Tokenizer,
        split: str = "train",
    ) -> RLPromptDataset:
        if split == "train":
            path = cfg.train_path
        elif split == "val":
            if cfg.val_path is None:
                raise ValueError(
                    "RLPromptDataset.from_config(split='val') requires "
                    "DataConfig.val_path to be set."
                )
            path = cfg.val_path
        else:
            raise ValueError(f"Unknown split: {split!r}")
        return cls(
            path=path,
            tokenizer=tokenizer,
            seq_len=cfg.seq_len,
            system_prompt=cfg.system_prompt,
        )
