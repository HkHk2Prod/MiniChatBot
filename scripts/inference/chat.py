"""Interactive console chat with a trained MiniChatBot model.

Multi-turn conversation: the full message history is re-rendered with
the chat template each turn, so the model sees prior context. Use this
against an SFT-trained model — raw pretrained checkpoints will produce
text continuation rather than chat-style replies.

Examples:
    # Auto-pick latest run's ckpt_best.pt under runs/
    python scripts/inference/chat.py

    # Filter by run_name
    python scripts/inference/chat.py --run-name sft_smoketest

    # Specific checkpoint or run dir (best-in-dir, then latest, fallback)
    python scripts/inference/chat.py --checkpoint runs/.../ckpt_best.pt

    # Sampling
    python scripts/inference/chat.py --strategy top_p --top-p 0.9 --temperature 0.8

    # System prompt
    python scripts/inference/chat.py --system "You are a terse, helpful assistant."

Slash commands (typed at the prompt):
    /quit, /exit       end the session
    /reset             clear conversation history (keeps system prompt)
    /system <text>     set/change the system prompt (clears history)
    /history           print the current conversation
    /help              show commands
"""

from __future__ import annotations

import argparse

import torch

from minichatbot.chat.template import render_prompt_for_completion
from minichatbot.inference.cli import (
    add_checkpoint_args,
    add_sampling_args,
    build_strategy,
    resolve_checkpoint,
    resolve_tokenizer_path,
)
from minichatbot.inference.generator import Generator
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.base import Tokenizer
from minichatbot.tokenizer.bpe import IM_END_TOKEN, BPETokenizer
from minichatbot.utils.torch_helpers import resolve_device

HELP_TEXT = (
    "Slash commands:\n"
    "  /quit, /exit       end the session\n"
    "  /reset             clear conversation history (keeps system prompt)\n"
    "  /system <text>     set/change the system prompt (clears history)\n"
    "  /history           print the current conversation\n"
    "  /help              show this message"
)


def truncate_history_to_fit(
    messages: list[dict],
    tokenizer: Tokenizer,
    max_prompt_tokens: int,
) -> list[dict]:
    """Drop oldest user/assistant pairs until the rendered prompt fits.

    Always preserves the system message (if present) and the most recent
    user turn. Returns a new list; never mutates the input.
    """
    msgs = list(messages)
    body_start = 1 if msgs and msgs[0]["role"] == "system" else 0
    while len(render_prompt_for_completion(msgs, tokenizer)) > max_prompt_tokens:
        # Need at least one non-system message (the latest user turn) to keep.
        if len(msgs) - body_start <= 1:
            return msgs
        del msgs[body_start]
        if len(msgs) > body_start and msgs[body_start]["role"] == "assistant":
            del msgs[body_start]
    return msgs


def handle_slash(
    user_input: str,
    messages: list[dict],
) -> tuple[bool, list[dict]]:
    """Process a /command. Returns (should_quit, new_messages).

    If the input wasn't a recognized command, prints an error and returns
    the unchanged messages list (caller should `continue`).
    """
    cmd, _, rest = user_input.partition(" ")
    cmd = cmd.lower()
    match cmd:
        case "/quit" | "/exit":
            return True, messages
        case "/help":
            print(HELP_TEXT)
            return False, messages
        case "/reset":
            sys_msg = messages[0] if messages and messages[0]["role"] == "system" else None
            new_messages = [sys_msg] if sys_msg else []
            print("(history cleared)")
            return False, new_messages
        case "/system":
            if not rest.strip():
                print("usage: /system <prompt text>")
                return False, messages
            print("(system prompt set; history cleared)")
            return False, [{"role": "system", "content": rest.strip()}]
        case "/history":
            if not messages:
                print("(empty)")
            else:
                for m in messages:
                    print(f"  [{m['role']}] {m['content']}")
            return False, messages
        case _:
            print(f"unknown command: {cmd}. /help for the list.")
            return False, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive console chat with a trained MiniChatBot model."
    )
    add_checkpoint_args(parser)
    add_sampling_args(
        parser,
        default_strategy="top_p",
        default_max_new_tokens=200,
        strategy_help=(
            "Sampling strategy. Default top_p — greedy on a small SFT'd model "
            "tends to fall into a fixed-point reply that repeats every turn."
        ),
    )
    parser.set_defaults(temperature=0.8)
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.5,
        help=(
            "Subtract f * count(token) from each token's logit (count includes "
            "prompt + generated). 0 disables. Default 0.5 — small SFT'd models "
            "loop without it. Range typically [0, 2]."
        ),
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.5,
        help=(
            "Flat one-shot subtraction from any token that has already appeared. "
            "Combine with --frequency-penalty for compounded effect. 0 disables."
        ),
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    ckpt_path = resolve_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        prefer_best=True,
    )
    print(f"checkpoint: {ckpt_path}")

    tokenizer_path = resolve_tokenizer_path(ckpt_path, args.tokenizer)
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"tokenizer:  {tokenizer_path} (vocab={tokenizer.vocab_size})")

    model = LanguageModel.load(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model:      {model.cfg.type} ({n_params / 1e6:.2f}M params) on {device}")

    # Chat mode requires the chat-end token. A pretrained-only tokenizer
    # without chat specials would generate forever (or until max tokens),
    # which isn't what the user expects from a chat REPL — fail loudly.
    stop_id = tokenizer.special_token_id(IM_END_TOKEN)
    if stop_id is None:
        raise SystemExit(
            "This tokenizer has no <|im_end|> token — it wasn't trained with "
            "chat specials. scripts/inference/chat.py requires a chat-tuned tokenizer "
            "(use scripts/inference/generate.py for raw text completion instead)."
        )

    strategy = build_strategy(
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    gen = Generator(
        strategy=strategy,
        eos_id=stop_id,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )

    # Reserve room in the context window for the assistant's reply.
    max_prompt_tokens = model.cfg.max_seq_len - args.max_new_tokens
    if max_prompt_tokens < 16:
        raise SystemExit(
            f"--max-new-tokens ({args.max_new_tokens}) leaves no room in "
            f"max_seq_len ({model.cfg.max_seq_len}). Lower --max-new-tokens."
        )

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"\nstrategy: {args.strategy} | max_new_tokens: {args.max_new_tokens}")
    print("Type your message and press enter. /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            should_quit, messages = handle_slash(user_input, messages)
            if should_quit:
                break
            continue

        messages.append({"role": "user", "content": user_input})
        messages = truncate_history_to_fit(messages, tokenizer, max_prompt_tokens)

        prompt_ids = render_prompt_for_completion(messages, tokenizer)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = gen.generate(model, prompt_tensor, max_new_tokens=args.max_new_tokens)
        new_ids = out[0].tolist()[len(prompt_ids) :]
        # Strip trailing <|im_end|> so the assistant's reply prints cleanly.
        if new_ids and new_ids[-1] == stop_id:
            new_ids = new_ids[:-1]
        reply = tokenizer.decode(new_ids, include_special=False)

        print(f"bot> {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
