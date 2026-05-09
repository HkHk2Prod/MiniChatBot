"""Chat-format primitives.

Centralized so SFT, RL, and inference share the same conversation
encoding rules. Right now this is just `render_messages` (ChatML-style
tokenization with assistant-only loss masking); future additions:
extracting the assistant's turn from a completion, validating message
schemas, alternative formats (Llama-3 headers, etc.).
"""

from minichatbot.chat.template import render_messages, render_prompt_for_completion

__all__ = ["render_messages", "render_prompt_for_completion"]
