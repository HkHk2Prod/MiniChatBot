"""Base class for tokenizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from minichatbot.config import TokenizerConfig


class Tokenizer(ABC):
    """Common interface for all tokenizers (BPE, SentencePiece, char, ...).

    Implementations wrap whichever backend is convenient (e.g., HF
    `tokenizers` for BPE) but expose the same shape to the rest of the
    codebase.
    """

    @abstractmethod
    def encode(self, text: str, include_special: bool = True) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int], include_special: bool = False) -> str: ...

    @abstractmethod
    def encode_batch(
        self, texts: list[str], include_special: bool = True
    ) -> list[list[int]]: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def pad_id(self) -> int: ...

    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @property
    @abstractmethod
    def bos_id(self) -> int | None: ...

    @abstractmethod
    def special_token_id(self, token: str) -> int | None: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Tokenizer: ...

    @classmethod
    def from_config(cls, cfg: TokenizerConfig) -> Tokenizer:
        """Build a tokenizer from config by loading from `cfg.path`.

        - `Tokenizer.from_config(cfg)` — registry-dispatched: reads
          `cfg.type` and loads via the matching tokenizer class.
        - `BPETokenizer.from_config(cfg)` — loads as BPE directly.
        """
        if cls is Tokenizer:
            # Lazy import to avoid the tokenizer/__init__.py <-> base.py cycle.
            from minichatbot.tokenizer import TOKENIZER_REGISTRY

            target_cls = TOKENIZER_REGISTRY[cfg.type]
        else:
            target_cls = cls
        return target_cls.load(cfg.path)
