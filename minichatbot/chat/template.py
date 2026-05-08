"""ChatML-format conversation rendering with assistant-only loss masking.

Layout per turn:
    <|im_start|>{role}\\n{content}<|im_end|>

Turns are separated by a single newline. No leading <bos> or trailing
newline by default (callers can prepend BOS if their setup wants it).

Loss-mask convention (returned in `labels`):
    - Tokens to LEARN (assistant content + the assistant's <|im_end|>)
      appear in `labels` at the position that predicts them.
    - Tokens NOT to learn (system, user, role tags, inter-turn newlines)
      become -100 in `labels` so PyTorch's cross_entropy(ignore_index=-100)
      skips them.
    - The final position has no next token to predict; its label is -100.

The pair (input_ids, labels) is already shifted: predicting `input_ids[i]`
should produce `labels[i]`. Loss code can call CE directly without further
shifting.
"""

from __future__ import annotations

from collections.abc import Iterable

from minichatbot.tokenizer.base import Tokenizer
from minichatbot.tokenizer.bpe import IM_END_TOKEN, IM_START_TOKEN

VALID_ROLES = {"system", "user", "assistant"}
IGNORE_INDEX = -100


def render_messages(
    messages: Iterable[dict[str, str]],
    tokenizer: Tokenizer,
) -> tuple[list[int], list[int]]:
    """Tokenize a conversation; return (input_ids, labels) ready for a
    decoder-only transformer with `cross_entropy(ignore_index=-100)`.

    `messages` is a list of `{"role": ..., "content": ...}` dicts. Roles
    must be in {"system", "user", "assistant"}; the loss is applied only
    to assistant content + its closing `<|im_end|>`.

    Returns:
        input_ids: list[int] of length L
        labels:    list[int] of length L (same length, -100 where masked)
    """
    im_start = tokenizer.special_token_id(IM_START_TOKEN)
    im_end = tokenizer.special_token_id(IM_END_TOKEN)
    if im_start is None or im_end is None:
        raise ValueError(
            "Tokenizer is missing chat tokens. Train the tokenizer with "
            "<|im_start|> and <|im_end|> in special_tokens (default in "
            "BPETokenizer.DEFAULT_SPECIALS)."
        )

    nl_ids = tokenizer.encode("\n", add_special=False)
    if not nl_ids:
        raise ValueError(
            "Tokenizer produced no tokens for '\\n' — this would break "
            "ChatML formatting. Check the tokenizer's byte-level alphabet."
        )

    # Build the full token sequence and a parallel mask. `is_target[i]` is
    # True iff token i should contribute to loss when it's the *target*
    # being predicted (i.e., when some earlier position predicts it).
    tokens: list[int] = []
    is_target: list[bool] = []

    for turn_idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        if role not in VALID_ROLES:
            raise ValueError(
                f"Unknown role {role!r}. Expected one of {sorted(VALID_ROLES)}."
            )

        # Inter-turn newline (skip before the first turn).
        if turn_idx > 0:
            for tid in nl_ids:
                tokens.append(tid)
                is_target.append(False)

        # Header: <|im_start|>{role}\n  -- always masked.
        tokens.append(im_start)
        is_target.append(False)
        for tid in tokenizer.encode(role, add_special=False):
            tokens.append(tid)
            is_target.append(False)
        for tid in nl_ids:
            tokens.append(tid)
            is_target.append(False)

        # Content: only assistant content contributes to loss.
        learn = role == "assistant"
        for tid in tokenizer.encode(content, add_special=False):
            tokens.append(tid)
            is_target.append(learn)

        # End-of-turn marker: assistant learns to emit it (so it stops);
        # for system/user this is a framework token, masked.
        tokens.append(im_end)
        is_target.append(learn)

    if len(tokens) < 2:
        raise ValueError("Empty conversation after rendering — need at least one turn.")

    # Shift: input_ids = tokens[:-1], labels[i] = tokens[i+1] if is_target[i+1]
    # else IGNORE_INDEX. The very last token has no next token, so it falls
    # off — we don't include it in input_ids.
    input_ids = tokens[:-1]
    labels = [
        tokens[i + 1] if is_target[i + 1] else IGNORE_INDEX
        for i in range(len(tokens) - 1)
    ]
    return input_ids, labels


def render_prompt_for_completion(
    messages: Iterable[dict[str, str]],
    tokenizer: Tokenizer,
) -> list[int]:
    """Render messages and append the assistant header for sampling.

    Output: token ids for `<turn1><turn2>...<|im_start|>assistant\\n`,
    leaving the model positioned to continue the assistant's response.
    The caller is expected to sample new tokens until `<|im_end|>` (or
    a max-tokens cap).

    Used at inference time and for SFT-aware training samples — anywhere
    you want the model to behave as a chat responder rather than a raw
    text continuer.
    """
    im_start = tokenizer.special_token_id(IM_START_TOKEN)
    im_end = tokenizer.special_token_id(IM_END_TOKEN)
    if im_start is None or im_end is None:
        raise ValueError(
            "Tokenizer is missing chat tokens. Train the tokenizer with "
            "<|im_start|> and <|im_end|> in special_tokens."
        )
    nl_ids = tokenizer.encode("\n", add_special=False)

    tokens: list[int] = []
    for turn_idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        if role not in VALID_ROLES:
            raise ValueError(
                f"Unknown role {role!r}. Expected one of {sorted(VALID_ROLES)}."
            )
        if turn_idx > 0:
            tokens.extend(nl_ids)
        tokens.append(im_start)
        tokens.extend(tokenizer.encode(role, add_special=False))
        tokens.extend(nl_ids)
        tokens.extend(tokenizer.encode(content, add_special=False))
        tokens.append(im_end)

    # Append the assistant header. After this point the model continues
    # in "assistant content" mode until it emits its own <|im_end|>.
    tokens.extend(nl_ids)
    tokens.append(im_start)
    tokens.extend(tokenizer.encode("assistant", add_special=False))
    tokens.extend(nl_ids)
    return tokens
