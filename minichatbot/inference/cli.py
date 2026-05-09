"""Shared CLI helpers for inference scripts (generate, chat, future REPLs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.utils.checkpoints import (
    find_best_checkpoint,
    find_best_checkpoint_in,
    find_latest_checkpoint,
    find_latest_checkpoint_in,
)


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Flags for locating a checkpoint, tokenizer, run dir, and device."""
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pt OR run dir. Default: auto-detect latest under --output-dir.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json. Default: read from checkpoint dir or saved config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="When auto-detecting, only consider runs ending with _{run_name}.",
    )
    parser.add_argument("--output-dir", default="runs", help="Where runs live (default: runs).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None)


def add_sampling_args(
    parser: argparse.ArgumentParser,
    *,
    default_strategy: str = "greedy",
    default_max_new_tokens: int = 100,
    strategy_help: str | None = None,
) -> None:
    """Flags for max_new_tokens and sampling strategy/params.

    Per-script tweaks for shared flag defaults (e.g. chat's lower temperature)
    are best applied via `parser.set_defaults(...)` after this call.
    """
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_max_new_tokens,
        help=f"Max tokens generated per turn (default: {default_max_new_tokens}).",
    )
    parser.add_argument(
        "--strategy",
        default=default_strategy,
        choices=sorted(SAMPLING_REGISTRY.keys()),
        help=strategy_help or f"Sampling strategy (default: {default_strategy}).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)


def resolve_checkpoint(
    *,
    checkpoint: str | None,
    output_dir: str,
    run_name: str | None,
    prefer_best: bool = False,
) -> Path:
    """Resolve a checkpoint flag value to a `.pt` path.

    `prefer_best=True` (chat) tries `ckpt_best.pt` first and falls back to
    latest. `prefer_best=False` (generate) only looks at latest. Either way,
    a directory is treated as a run dir.
    """
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.is_file():
            return p
        if p.is_dir():
            ckpt = (find_best_checkpoint(p) or find_latest_checkpoint(p)) if prefer_best else find_latest_checkpoint(p)
            if ckpt is None:
                raise SystemExit(f"No checkpoints found in {p}/checkpoints/")
            return ckpt
        raise SystemExit(f"Checkpoint path does not exist: {p}")

    ckpt = None
    if prefer_best:
        ckpt = find_best_checkpoint_in(output_dir, run_name=run_name)
    if ckpt is None:
        ckpt = find_latest_checkpoint_in(output_dir, run_name=run_name)
    if ckpt is None:
        msg = f"No checkpoints found under {output_dir}"
        if run_name:
            msg += f" matching run_name='{run_name}'"
        msg += ". Pass --checkpoint explicitly, or train a model first."
        raise SystemExit(msg)
    return ckpt


def resolve_tokenizer_path(checkpoint_path: Path, override: str | None) -> str:
    """Find tokenizer.json: explicit override -> sibling of run dir -> saved config."""
    if override is not None:
        return override
    # Preferred: the tokenizer copy that the runner writes alongside the run
    # dir. This makes the run self-contained — the original data/ path may
    # not even exist anymore (different machine, deleted, renamed).
    # Layout: runs/<ts>_<name>/checkpoints/ckpt.pt -> ../tokenizer.json
    run_dir_tokenizer = checkpoint_path.parent.parent / "tokenizer.json"
    if run_dir_tokenizer.exists():
        return str(run_dir_tokenizer)
    # Fallback: whatever the saved config points to. Older runs (before the
    # run-dir-tokenizer convention) or hand-built checkpoints land here.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_config = state.get("full_config")
    if full_config is None:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} has no 'full_config' and no "
            f"sibling tokenizer.json. Pass --tokenizer explicitly."
        )
    path = full_config.get("tokenizer", {}).get("path")
    if path is None:
        raise SystemExit(
            f"Checkpoint at {checkpoint_path} has no tokenizer.path in its "
            "saved config. Pass --tokenizer explicitly."
        )
    return path


def build_strategy(
    *,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
):
    cls = SAMPLING_REGISTRY[strategy]
    if strategy == "greedy":
        return cls()
    if strategy == "temperature":
        return cls(temperature=temperature)
    if strategy == "top_k":
        return cls(k=top_k, temperature=temperature)
    if strategy == "top_p":
        return cls(p=top_p, temperature=temperature)
    raise SystemExit(f"Unsupported strategy: {strategy}")
