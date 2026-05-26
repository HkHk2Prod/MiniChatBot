"""Tests for ChatML rendering and loss masking (minichatbot/chat/template.py)."""

from __future__ import annotations

import pytest
from _stubs import StubTokenizer

from minichatbot.chat.template import (
    IGNORE_INDEX,
    render_messages,
    render_prompt_for_completion,
)
from minichatbot.tokenizer.bpe import IM_END_TOKEN, IM_START_TOKEN


def _learned_targets(
    messages: list[dict[str, str]], tok: StubTokenizer
) -> list[int]:
    """What `render_messages` *should* learn: assistant content + its <|im_end|>,
    concatenated across assistant turns, in order."""
    im_end = tok.special_token_id(IM_END_TOKEN)
    out: list[int] = []
    for msg in messages:
        if msg["role"] == "assistant":
            out.extend(tok.encode(msg["content"], include_special=False))
            out.append(im_end)
    return out


def test_input_ids_and_labels_equal_length(stub_tokenizer: StubTokenizer) -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    input_ids, labels = render_messages(messages, stub_tokenizer)
    assert len(input_ids) == len(labels)
    assert len(input_ids) > 0


def test_only_assistant_content_and_im_end_are_learned(
    stub_tokenizer: StubTokenizer,
) -> None:
    # Distinct content so any leakage from system/user would be obvious.
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "ANSWER"},
        {"role": "user", "content": "AGAIN"},
        {"role": "assistant", "content": "REPLY"},
    ]
    _, labels = render_messages(messages, stub_tokenizer)

    learned = [tid for tid in labels if tid != IGNORE_INDEX]
    assert learned == _learned_targets(messages, stub_tokenizer)


def test_labels_are_pre_shifted(stub_tokenizer: StubTokenizer) -> None:
    # labels[i] predicts the token at input_ids[i+1] (or -100 if masked).
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    input_ids, labels = render_messages(messages, stub_tokenizer)
    for i in range(len(labels) - 1):
        assert labels[i] == IGNORE_INDEX or labels[i] == input_ids[i + 1]


def test_inter_turn_and_header_newlines_are_masked(
    stub_tokenizer: StubTokenizer,
) -> None:
    # No newlines in any content, so the only '\n' tokens come from headers and
    # inter-turn separators — all of which must be masked.
    newline_id = stub_tokenizer.encode("\n", include_special=False)[0]
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    _, labels = render_messages(messages, stub_tokenizer)
    assert newline_id not in [tid for tid in labels if tid != IGNORE_INDEX]


def test_assistant_only_learns_nothing_from_other_roles(
    stub_tokenizer: StubTokenizer,
) -> None:
    messages = [
        {"role": "system", "content": "ZZZ"},
        {"role": "user", "content": "YYY"},
    ]
    # No assistant turn -> nothing should be learned.
    _, labels = render_messages(messages, stub_tokenizer)
    assert all(tid == IGNORE_INDEX for tid in labels)


def test_render_messages_unknown_role_raises(stub_tokenizer: StubTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown role"):
        render_messages([{"role": "tool", "content": "x"}], stub_tokenizer)


def test_render_messages_empty_conversation_raises(
    stub_tokenizer: StubTokenizer,
) -> None:
    with pytest.raises(ValueError, match="Empty conversation"):
        render_messages([], stub_tokenizer)


def test_render_messages_missing_chat_tokens_raises() -> None:
    tok = StubTokenizer(with_chat_tokens=False)
    with pytest.raises(ValueError, match="missing chat tokens"):
        render_messages([{"role": "user", "content": "hi"}], tok)


def test_render_prompt_ends_with_assistant_header(
    stub_tokenizer: StubTokenizer,
) -> None:
    messages = [{"role": "user", "content": "hi"}]
    tokens = render_prompt_for_completion(messages, stub_tokenizer)

    im_start = stub_tokenizer.special_token_id(IM_START_TOKEN)
    suffix = (
        [im_start]
        + stub_tokenizer.encode("assistant", include_special=False)
        + stub_tokenizer.encode("\n", include_special=False)
    )
    assert tokens[-len(suffix):] == suffix


def test_render_prompt_unknown_role_raises(stub_tokenizer: StubTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown role"):
        render_prompt_for_completion([{"role": "tool", "content": "x"}], stub_tokenizer)


def test_render_prompt_missing_chat_tokens_raises() -> None:
    tok = StubTokenizer(with_chat_tokens=False)
    with pytest.raises(ValueError, match="missing chat tokens"):
        render_prompt_for_completion([{"role": "user", "content": "hi"}], tok)
