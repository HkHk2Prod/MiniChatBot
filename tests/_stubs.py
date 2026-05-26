"""Lightweight stand-ins for the Tokenizer and LanguageModel interfaces.

These deliberately avoid the real transformer and BPE backend so the unit
suite stays fast and CPU-only. Each stub implements just enough of its base
interface to drive the logic under test with hand-checkable values.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from minichatbot.model.base import ModelOutput
from minichatbot.tokenizer.base import Tokenizer
from minichatbot.tokenizer.bpe import (
    BOS_TOKEN,
    EOS_TOKEN,
    IM_END_TOKEN,
    IM_START_TOKEN,
    PAD_TOKEN,
)


class StubTokenizer(Tokenizer):
    """Deterministic char-level tokenizer implementing the Tokenizer interface.

    Regular text encodes to one id per character (``id = _CHAR_BASE + ord(ch)``)
    so the resulting ids are trivially predictable; a small fixed set of named
    special tokens get low ids. This lets the chat-template tests assert on
    exact id sequences without training a real BPE model. Pass
    ``with_chat_tokens=False`` to model a tokenizer that lacks the ChatML
    markers (so ``special_token_id`` returns ``None`` for them).
    """

    _CHAR_BASE = 256
    _SPECIAL_IDS = {
        PAD_TOKEN: 0,
        BOS_TOKEN: 1,
        EOS_TOKEN: 2,
        IM_START_TOKEN: 3,
        IM_END_TOKEN: 4,
    }

    def __init__(self, *, with_chat_tokens: bool = True) -> None:
        self._specials = dict(self._SPECIAL_IDS)
        if not with_chat_tokens:
            del self._specials[IM_START_TOKEN]
            del self._specials[IM_END_TOKEN]
        self._id_to_special = {v: k for k, v in self._specials.items()}

    def encode(self, text: str, include_special: bool = True) -> list[int]:
        return [self._CHAR_BASE + ord(c) for c in text]

    def decode(self, ids: list[int], include_special: bool = False) -> str:
        out: list[str] = []
        for i in ids:
            if i in self._id_to_special:
                if include_special:
                    out.append(self._id_to_special[i])
            else:
                out.append(chr(i - self._CHAR_BASE))
        return "".join(out)

    def encode_batch(
        self, texts: list[str], include_special: bool = True
    ) -> list[list[int]]:
        return [self.encode(t, include_special) for t in texts]

    @property
    def vocab_size(self) -> int:
        # Enough headroom for every unicode code point plus the specials.
        return self._CHAR_BASE + 0x110000

    @property
    def pad_id(self) -> int:
        return self._SPECIAL_IDS[PAD_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._SPECIAL_IDS[EOS_TOKEN]

    @property
    def bos_id(self) -> int | None:
        return self._SPECIAL_IDS[BOS_TOKEN]

    def special_token_id(self, token: str) -> int | None:
        return self._specials.get(token)

    def save(self, path: str | Path) -> None:  # pragma: no cover - unused by tests
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> StubTokenizer:  # pragma: no cover - unused
        raise NotImplementedError


class StubLogitsModel(nn.Module):
    """Position-wise causal stand-in for a LanguageModel.

    ``logits[b, t] = table[input_ids[b, t]]`` — each position's logits depend
    only on the token *at* that position. That makes the model trivially
    causal and invariant to right-padding, exactly the properties
    :func:`minichatbot.eval.scoring.score_batch` relies on, so a single table
    drives deterministic, hand-checkable scores. ``table[i]`` is the
    next-token logit row emitted when the current token is ``i``.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table.float())

    @property
    def vocab_size(self) -> int:
        return int(self.table.size(-1))

    def forward(self, input_ids: torch.Tensor, state: object | None = None) -> ModelOutput:
        return ModelOutput(logits=self.table[input_ids])

    def init_state(self, batch_size: int, device: torch.device) -> None:  # pragma: no cover
        return None
