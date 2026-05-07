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
    def encode(self, text: str, add_special: bool = True) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int], skip_special: bool = True) -> str: ...

    @abstractmethod
    def encode_batch(
        self, texts: list[str], add_special: bool = True
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
        """Build a tokenizer from config.

        - `Tokenizer.from_config(cfg)` — registry-dispatched: reads
          `cfg.type` and instantiates the matching tokenizer class.
        - `BPETokenizer.from_config(cfg)` (concrete subclass) — builds
          directly. Subclasses must override this method.
        """
        if cls is Tokenizer:
            # Lazy import to avoid the tokenizer/__init__.py <-> base.py cycle.
            from minichatbot.tokenizer import TOKENIZER_REGISTRY

            target_cls = TOKENIZER_REGISTRY[cfg.type]
            return target_cls.from_config(cfg)
        raise NotImplementedError(
            f"{cls.__name__} must override from_config"
        )
