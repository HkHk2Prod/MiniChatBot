"""Reconcile YAML model config with a checkpoint's saved model_config.

When loading a checkpoint into a model built from a YAML, the two configs
must agree on shape-affecting fields or `load_state_dict` fails with
cryptic size-mismatch errors. This module checks compatibility up front
and either:

- For ``--resume``: takes the checkpoint's shape fields as authoritative
  (you're continuing the same model — the saved arch is the truth),
  warning loudly that the YAML's values were overridden.
- For ``--from-pretrained``: treats shape mismatches as fatal and prints
  a clear diff with a suggested fix, since a new training trajectory on
  a different arch isn't what the user meant.

Non-shape-affecting field changes (dropout, max_seq_len) never fail —
they print a warning and proceed with the YAML's value.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import fields
from typing import Literal

from minichatbot.config import ModelConfig

# Fields whose values determine parameter tensor shapes in the saved
# state_dict. A mismatch on any of these will make load_state_dict fail.
# - `n_heads`: technically doesn't change weight shapes (Q/K/V projections
#   are always d_model -> d_model), but it changes the attention reshape
#   and head_dim semantics, so a silent mismatch would produce a broken
#   model. Treat as shape-affecting for safety.
# - `max_seq_len` is intentionally NOT here: RoPE caches are registered
#   with persistent=False, so they're not in state_dict and won't trigger
#   load failures.
# - `dropout` is runtime-only (no weights), so it never affects loading.
_SHAPE_FIELDS: frozenset[str] = frozenset({
    "type",
    "vocab_size",
    "n_layers",
    "n_heads",
    "d_model",
    "d_ff",
    "tie_embeddings",
    "norm_type",
})


def parse_ckpt_model_config(raw: dict[str, object]) -> ModelConfig:
    """Build a ModelConfig from a checkpoint's saved dict.

    Filters to known fields so a checkpoint written with an older or
    newer ModelConfig schema (extra/missing keys) still loads — missing
    fields get dataclass defaults, unknown fields are silently dropped.
    """
    known = {f.name for f in fields(ModelConfig)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return ModelConfig(**filtered)  # type: ignore[arg-type]


def reconcile_model_config(
    yaml_cfg: ModelConfig,
    ckpt_cfg: ModelConfig,
    *,
    mode: Literal["resume", "from_pretrained"],
) -> tuple[ModelConfig, list[str]]:
    """Compare and resolve YAML vs checkpoint model configs.

    Returns ``(effective_config, warning_banners)``. The warnings are
    intended to be replayed inside `Trainer.fit()` (after the LogFile
    callback installs its stdout tee) so they end up in the per-run
    `log.txt`, not just the console.

    Raises ``SystemExit`` with a clear diff if ``mode == "from_pretrained"``
    and any shape-affecting field disagrees.
    """
    diffs: list[tuple[str, object, object]] = [
        (f.name, getattr(yaml_cfg, f.name), getattr(ckpt_cfg, f.name))
        for f in fields(ModelConfig)
        if getattr(yaml_cfg, f.name) != getattr(ckpt_cfg, f.name)
    ]
    if not diffs:
        return yaml_cfg, []

    shape_diffs = [d for d in diffs if d[0] in _SHAPE_FIELDS]
    other_diffs = [d for d in diffs if d[0] not in _SHAPE_FIELDS]

    warnings: list[str] = []

    if shape_diffs:
        if mode == "from_pretrained":
            # Fatal: load_state_dict would error a moment later anyway.
            # Print to stderr now (the LogFile callback isn't up yet) and
            # exit before we waste time building a model that can't load.
            print(_fatal_banner(shape_diffs), file=sys.stderr, flush=True)
            raise SystemExit(2)
        # mode == "resume": override the YAML with checkpoint values.
        warnings.append(_resume_override_banner(shape_diffs))

    effective = dataclasses.replace(yaml_cfg)
    if mode == "resume" and shape_diffs:
        for name, _, ckpt_val in shape_diffs:
            setattr(effective, name, ckpt_val)

    if other_diffs:
        warnings.append(_behavioral_diff_banner(other_diffs))

    return effective, warnings


def _banner(lines: list[str]) -> str:
    width = max((len(ln) for ln in lines), default=0)
    border = "+" + "=" * (width + 2) + "+"
    body = "\n".join(f"| {ln.ljust(width)} |" for ln in lines)
    return "\n".join(["", border, body, border, ""])


def _fmt_table(
    rows: list[tuple[str, ...]],
    headers: tuple[str, ...],
) -> list[str]:
    str_rows = [tuple(str(x) for x in r) for r in rows]
    cols = list(zip(headers, *str_rows))
    widths = [max(len(c) for c in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers)]
    out.append("  ".join("-" * w for w in widths))
    for r in str_rows:
        out.append(fmt.format(*r))
    return out


def _fatal_banner(diffs: list[tuple[str, object, object]]) -> str:
    lines = [
        "FATAL: --from-pretrained: YAML model config / checkpoint mismatch",
        "",
        "The YAML's model: block has shape-affecting fields that don't match",
        "the checkpoint's saved architecture. load_state_dict would fail with",
        "size mismatches; aborting now with a clearer message instead.",
        "",
    ]
    lines.extend(_fmt_table(
        rows=[(n, repr(y), repr(c)) for n, y, c in diffs],
        headers=("field", "YAML", "checkpoint"),
    ))
    lines.extend([
        "",
        "Fix one of:",
        "  - Align the YAML's model: block with the checkpoint's saved arch",
        "  - Pick a checkpoint trained with this YAML's architecture",
    ])
    return _banner(lines)


def _resume_override_banner(diffs: list[tuple[str, object, object]]) -> str:
    lines = [
        "WARNING: --resume: overriding YAML model config with checkpoint values",
        "",
        "These shape-affecting fields differ between the YAML and the saved",
        "checkpoint. Since you're resuming the same run, the checkpoint is",
        "authoritative — building the model with the checkpoint's values:",
        "",
    ]
    lines.extend(_fmt_table(
        rows=[(n, repr(y), repr(c)) for n, y, c in diffs],
        headers=("field", "YAML (ignored)", "checkpoint (used)"),
    ))
    return _banner(lines)


def _behavioral_diff_banner(diffs: list[tuple[str, object, object]]) -> str:
    lines = [
        "WARNING: YAML model config differs from checkpoint (non-shape fields)",
        "",
        "These don't affect loading but change runtime behavior. Proceeding",
        "with YAML values — the checkpoint's values are shown for reference:",
        "",
    ]
    lines.extend(_fmt_table(
        rows=[(n, repr(y), repr(c)) for n, y, c in diffs],
        headers=("field", "YAML (used)", "checkpoint"),
    ))
    return _banner(lines)
