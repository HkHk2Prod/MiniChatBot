"""LogFile callback: mirrors stdout + stderr to a per-run text transcript.

The transcript is byte-for-byte equivalent to what you saw on the
terminal — useful for `tail -f` over SSH during a run, scrolling
through after the fact, or grepping. JSONL/TensorBoard cover the
machine-readable side; this covers the human-readable side.

Order matters: place this callback FIRST in the YAML callbacks list so
the tee is active before any other callback (e.g. ConsoleCallback)
prints in `on_train_start`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Any

from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext


class _Tee:
    """Stream proxy that fans writes out to multiple destinations."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


@CALLBACK_REGISTRY.register("logfile")
class LogFileCallback(Callback):
    """Tee stdout + stderr to {run_dir}/{filename} during fit().

    Captures everything other callbacks print, plus any tracebacks
    raised inside the training loop. The original stdout/stderr are
    restored in on_train_end even if training crashed.
    """

    def __init__(self, filename: str = "log.txt") -> None:
        self.filename = filename
        self._fh: IO[str] | None = None
        self._original_stdout: Any = None
        self._original_stderr: Any = None

    def on_train_start(self, ctx: CallbackContext) -> None:
        path = Path(ctx.run_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1, encoding="utf-8")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _Tee(self._original_stdout, self._fh)
        sys.stderr = _Tee(self._original_stderr, self._fh)

    def on_train_end(self, ctx: CallbackContext) -> None:
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
            self._original_stderr = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
