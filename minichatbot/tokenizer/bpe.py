"""Byte-level BPE tokenizer wrapping the HuggingFace `tokenizers` backend.

Byte-level BPE (GPT-2 style) is the default because it can encode any input
without an UNK fallback and round-trips arbitrary bytes losslessly.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers

from minichatbot.tokenizer import TOKENIZER_REGISTRY
from minichatbot.tokenizer.base import Tokenizer

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
NUM_RESERVED_TOKENS = 8
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(NUM_RESERVED_TOKENS)]
DEFAULT_SPECIALS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    *RESERVED_TOKENS,
]


@TOKENIZER_REGISTRY.register("bpe")
class BPETokenizer(Tokenizer):
    def __init__(self, hf: HFTokenizer) -> None:
        self._hf = hf
        self._pad_id = self._require(PAD_TOKEN)
        self._eos_id = self._require(EOS_TOKEN)
        self._bos_id = hf.token_to_id(BOS_TOKEN)

    def _require(self, token: str) -> int:
        tid = self._hf.token_to_id(token)
        if tid is None:
            raise ValueError(
                f"BPETokenizer requires special token {token!r}; "
                f"retrain with it in special_tokens."
            )
        return tid

    def encode(self, text: str, include_special: bool = True) -> list[int]:
        return self._hf.encode(text, add_special_tokens=include_special).ids

    def decode(self, ids: list[int], include_special: bool = False) -> str:
        return self._hf.decode(ids, skip_special_tokens=not include_special)

    def encode_batch(
        self, texts: list[str], include_special: bool = True
    ) -> list[list[int]]:
        encoded = self._hf.encode_batch(texts, add_special_tokens=include_special)
        return [e.ids for e in encoded]

    @property
    def vocab_size(self) -> int:
        return self._hf.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def bos_id(self) -> int | None:
        return self._bos_id

    def special_token_id(self, token: str) -> int | None:
        return self._hf.token_to_id(token)

    def save(self, path: str | Path) -> None:
        self._hf.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        return cls(HFTokenizer.from_file(str(path)))

    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> BPETokenizer:
        specials = special_tokens if special_tokens is not None else DEFAULT_SPECIALS

        hf = HFTokenizer(models.BPE())
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        hf.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=show_progress,
        )
        hf.train_from_iterator(corpus, trainer=trainer)
        tok = cls(hf)
        hf.post_processor = processors.TemplateProcessing(
            single=f"$A {EOS_TOKEN}",
            special_tokens=[(EOS_TOKEN, tok._eos_id)],
        )
        return tok
