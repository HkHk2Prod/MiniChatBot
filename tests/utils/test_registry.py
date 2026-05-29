"""Tests for the generic Registry (minichatbot/utils/registry.py)."""

from __future__ import annotations

import pytest

from minichatbot.utils.registry import Registry


def test_register_and_getitem() -> None:
    reg: Registry[object] = Registry("widget")

    @reg.register("alpha")
    class Alpha:
        pass

    assert reg["alpha"] is Alpha
    assert "alpha" in reg
    assert "missing" not in reg


def test_duplicate_key_raises_value_error() -> None:
    reg: Registry[object] = Registry("widget")

    @reg.register("dup")
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("dup")
        class Second:
            pass


def test_missing_key_raises_keyerror_with_available_listing() -> None:
    reg: Registry[object] = Registry("widget")

    @reg.register("alpha")
    class Alpha:
        pass

    with pytest.raises(KeyError) as excinfo:
        reg["nope"]

    message = str(excinfo.value)
    assert "no entry 'nope'" in message
    assert "Available:" in message
    assert "alpha" in message  # the listing surfaces what *is* registered


def test_keys_returns_sorted_list() -> None:
    reg: Registry[object] = Registry("widget")
    for key in ("charlie", "alpha", "bravo"):

        @reg.register(key)
        class _Entry:
            pass

    assert reg.keys() == ["alpha", "bravo", "charlie"]


def test_repr_lists_entries() -> None:
    reg: Registry[object] = Registry("widget")

    @reg.register("alpha")
    class Alpha:
        pass

    assert repr(reg) == "Registry('widget', entries=['alpha'])"


def test_str_empty_registry() -> None:
    reg: Registry[object] = Registry("widget")
    assert str(reg) == "widget registry (empty)"


def test_str_non_empty_registry() -> None:
    reg: Registry[object] = Registry("widget")

    @reg.register("alpha")
    class Alpha:
        pass

    @reg.register("beta")
    class Beta:
        pass

    text = str(reg)
    assert text.startswith("widget registry (2 entries):")
    assert "->" in text  # each line maps key -> class name
    assert "alpha" in text and "beta" in text
    assert "Alpha" in text and "Beta" in text


def test_str_singular_entry_count() -> None:
    reg: Registry[object] = Registry("widget")

    @reg.register("only")
    class Only:
        pass

    assert str(reg).startswith("widget registry (1 entry):")
