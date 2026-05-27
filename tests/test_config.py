"""Tests for config parsing, round-tripping, and validation (minichatbot/config.py)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from minichatbot.config import (
    CallbackSpec,
    Config,
    DataConfig,
    ModelConfig,
    load_config,
    save_config,
)


def _valid_cfg_dict() -> dict[str, Any]:
    """A minimal config that passes every `validate` invariant."""
    return {
        "run_name": "unit-test",
        "device": "cpu",
        "data": {"train_path": "data/train.bin", "seq_len": 128},
        "model": {"max_seq_len": 256, "d_model": 64, "n_heads": 8},
        "tokenizer": {"path": "tok.json"},
        "optim": {"lr_schedule": "cosine", "betas": [0.9, 0.95]},
        "trainer": {"batch_size": 4, "grad_accum_steps": 1, "precision": "bf16"},
        "rl": {"group_size": 4, "top_p": 0.9},
        "callbacks": [{"type": "console", "params": {"every": 10}}],
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_coerces_nested_dataclasses_and_tuples(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(tmp_path / "c.yaml", _valid_cfg_dict()))

    assert isinstance(cfg, Config)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.data, DataConfig)
    # list-in-yaml -> tuple field, with element coercion.
    assert cfg.optim.betas == (0.9, 0.95)
    assert isinstance(cfg.optim.betas, tuple)
    # list[CallbackSpec] coercion.
    assert isinstance(cfg.callbacks[0], CallbackSpec)
    assert cfg.callbacks[0].type == "console"
    assert cfg.callbacks[0].params == {"every": 10}


def test_save_load_round_trip(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(tmp_path / "c.yaml", _valid_cfg_dict()))
    out = tmp_path / "out.yaml"
    save_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded == cfg


def test_to_yaml_from_yaml_helpers_round_trip(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(tmp_path / "c.yaml", _valid_cfg_dict()))
    out = tmp_path / "via_methods.yaml"
    cfg.to_yaml(out)
    assert Config.from_yaml(out) == cfg


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(bad)


def test_unknown_field_raises(tmp_path: Path) -> None:
    data = _valid_cfg_dict()
    data["model"]["bogus_field"] = 123
    with pytest.raises(ValueError, match="Unknown fields"):
        load_config(_write_yaml(tmp_path / "c.yaml", data))


# Each mutation breaks exactly one validated invariant; load_config -> validate
# must reject it. The id labels document which invariant is exercised.
_INVALID_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "precision": lambda d: d["trainer"].__setitem__("precision", "int8"),
    "norm_type": lambda d: d["model"].__setitem__("norm_type", "groupnorm"),
    "lr_schedule": lambda d: d["optim"].__setitem__("lr_schedule", "exponential"),
    "device": lambda d: d.__setitem__("device", "tpu"),
    "d_model_not_divisible": lambda d: d["model"].__setitem__("d_model", 65),
    "batch_size_zero": lambda d: d["trainer"].__setitem__("batch_size", 0),
    "grad_accum_zero": lambda d: d["trainer"].__setitem__("grad_accum_steps", 0),
    "seq_len_exceeds_max": lambda d: d["data"].__setitem__("seq_len", 9999),
    "group_size_too_small": lambda d: d["rl"].__setitem__("group_size", 1),
    "top_p_zero": lambda d: d["rl"].__setitem__("top_p", 0.0),
    "top_p_above_one": lambda d: d["rl"].__setitem__("top_p", 1.5),
}


@pytest.mark.parametrize(
    "mutate", _INVALID_MUTATIONS.values(), ids=list(_INVALID_MUTATIONS.keys())
)
def test_validate_rejects_bad_constraints(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    data = _valid_cfg_dict()
    mutate(data)
    with pytest.raises(ValueError):
        load_config(_write_yaml(tmp_path / "c.yaml", data))


def test_valid_config_passes_validation(tmp_path: Path) -> None:
    # Sanity check that the baseline really is valid (so the negatives above
    # fail for the mutated reason, not a latent baseline error).
    cfg = load_config(_write_yaml(tmp_path / "c.yaml", _valid_cfg_dict()))
    assert cfg.run_name == "unit-test"
