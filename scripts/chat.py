"""Interactive console chat with a trained MiniChatBot model.

Multi-turn conversation: the full message history is re-rendered with
the chat template each turn, so the model sees prior context. Use this
against an SFT-trained model — raw pretrained checkpoints will produce
text continuation rather than chat-style replies.

Examples:
    # Auto-pick latest run's ckpt_best.pt under runs/
    python scripts/chat.py

    # Filter by run_name
    python scripts/chat.py --run-name sft_smoketest

    # Specific checkpoint or run dir (best-in-dir, then latest, fallback)
    python scripts/chat.py --checkpoint runs/.../ckpt_best.pt

    # Sampling
    python scripts/chat.py --strategy top_p --top-p 0.9 --temperature 0.8

    # System prompt
    python scripts/chat.py --system "You are a terse, helpful assistant."

Slash commands (typed at the prompt):
    /quit, /exit       end the session
    /reset             clear conversation history (keeps system prompt)
    /system <text>     set/change the system prompt (clears history)
    /history           print the current conversation
    /help              show commands
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minichatbot.chat.template import render_prompt_for_completion
from minichatbot.inference.generator import Generator
from minichatbot.inference.strategies import SAMPLING_REGISTRY
from minichatbot.model.base import LanguageModel
from minichatbot.tokenizer.base import Tokenizer
from minichatbot.tokenizer.bpe import IM_END_TOKEN, BPETokenizer
from minichatbot.utils.checkpoints import (
    find_best_checkpoint,
    find_best_checkpoint_in,
    find_latest_checkpoint,
    find_latest_checkpoint_in,
)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    """Resolve --checkpoint to a .pt path. Prefers ckpt_best.pt over latest.

    Chat is post-training — the user wants to talk to the model that
    generalized best, not whatever fell out of the final step.
    """
    if args.checkpoint is not None:
        p = Path(args.checkpoint)
        if p.is_file():
            return p
        if p.is_dir():
            ckpt = find_best_checkpoint(p) or find_latest_checkpoint(p)
            if ckpt is None:
                raise SystemExit(f"No checkpoints found in {p}/checkpoints/")
            return ckpt
        raise SystemExit(f"Checkpoint path does not exist: {p}")

    ckpt = find_best_checkpoint_in(args.output_dir, run_name=args.run_name)
    if ckpt is None:
        ckpt = find_latest_checkpoint_in(args.output_dir, run_name=args.run_name)
    if ckpt is None:
        msg = f"No checkpoints found under {args.output_dir}"
        if args.run_name:
            msg += f" matching run_name='{args.run_name}'"
        msg += ". Pass --checkpoint explicitly, or train a model first."
        raise SystemExit(msg)
    return ckpt


def resolve_tokenizer_path(checkpoint_path: Path, override: str | None) -> str:
    if override is not None:
        return override
    run_dir_tokenizer = checkpoint_path.parent.parent / "tokenizer.json"
    if run_dir_tokenizer.exists():
        return str(run_dir_tokenizer)
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


def build_strategy(args: argparse.Namespace):
    cls = SAMPLING_REGISTRY[args.strategy]
    if args.strategy == "greedy":
        return cls()
    if args.strategy == "temperature":
        return cls(temperature=args.temperature)
    if args.strategy == "top_k":
        return cls(k=args.top_k, temperature=args.temperature)
    if args.strategy == "top_p":
        return cls(p=args.top_p, temperature=args.temperature)
    raise SystemExit(f"Unsupported strategy: {args.strategy}")


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
    if cmd in ("/quit", "/exit"):
        return True, messages
    if cmd == "/help":
        print(HELP_TEXT)
        return False, messages
    if cmd == "/reset":
        sys_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        new_messages = [sys_msg] if sys_msg else []
        print("(history cleared)")
        return False, new_messages
    if cmd == "/system":
        if not rest.strip():
            print("usage: /system <prompt text>")
            return False, messages
        print("(system prompt set; history cleared)")
        return False, [{"role": "system", "content": rest.strip()}]
    if cmd == "/history":
        if not messages:
            print("(empty)")
        else:
            for m in messages:
                print(f"  [{m['role']}] {m['content']}")
        return False, messages
    print(f"unknown command: {cmd}. /help for the list.")
    return False, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive console chat with a trained MiniChatBot model."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pt OR run dir. Default: latest run's ckpt_best.pt under --output-dir.",
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
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max tokens generated per assistant turn (default: 200).",
    )
    parser.add_argument(
        "--strategy",
        default="greedy",
        choices=sorted(SAMPLING_REGISTRY.keys()),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    ckpt_path = resolve_checkpoint(args)
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
            "chat specials. scripts/chat.py requires a chat-tuned tokenizer "
            "(use scripts/generate.py for raw text completion instead)."
        )

    strategy = build_strategy(args)
    gen = Generator(strategy=strategy, eos_id=stop_id)

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
        new_ids = out[0].tolist()[len(prompt_ids):]
        # Strip trailing <|im_end|> so the assistant's reply prints cleanly.
        if new_ids and new_ids[-1] == stop_id:
            new_ids = new_ids[:-1]
        reply = tokenizer.decode(new_ids, skip_special=True)

        print(f"bot> {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
