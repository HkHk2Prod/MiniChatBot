"""Reusable benchmark engine: per-phase prompt sets → completions on disk.

This is the library form of `scripts/inference/benchmark.py`. The CLI
script is a thin wrapper that handles checkpoint resolution + model
loading and then calls `run_benchmark` here. The same function is also
called from `BenchmarkCallback` at the end of training so every run
produces a `benchmark_<phase>.txt` next to its checkpoints.

The benchmark YAML splits prompts into three groups per phase:

    near       — prompts very close to (or verbatim from) the training
                 corpus. A working checkpoint should produce ~perfect
                 output here.
    generalize — novel content inside the same domain / format. The
                 real test: did training generalize, or memorize?
    ood        — outside the corpus entirely. Failures expected; useful
                 for catastrophic-forgetting probes (rl after SFT).

For phase=rl, `near` and `generalize` carry numeric `reference` answers
and are scored by `GSM8KReward`; `ood` is plain text and unscored.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import torch
import yaml

from minichatbot.chat.template import render_prompt_for_completion
from minichatbot.inference.generator import Generator
from minichatbot.inference.text_generator import TextGenerator
from minichatbot.model.base import LanguageModel
from minichatbot.rl.rewards.gsm8k import GSM8KReward, extract_final_answer
from minichatbot.tokenizer.base import Tokenizer


PHASES = ("pretrain", "sft", "rl")
GROUPS = ("near", "generalize", "ood")
GROUP_HEADERS: dict[str, str] = {
    "near": "near — prompts very close to (or verbatim from) the training data",
    "generalize": "generalize — novel content inside the same domain / format",
    "ood": "ood — outside the training data; weakness is expected here",
}


def load_phase_prompts(path: Path | str, phase: str) -> dict[str, Any]:
    """Read the benchmark YAML and return the entry for one phase.

    Raises ValueError if the file doesn't list the requested phase —
    callers (CLI / callback) translate that into their own user-facing
    error.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if phase not in data:
        raise ValueError(
            f"benchmark prompts file {p} has no entry for phase {phase!r}. "
            f"Found phases: {sorted(data.keys())}."
        )
    return data[phase]


def _group_prompts(phase_cfg: dict[str, Any], group: str) -> list[Any]:
    """Return the prompt list for one group, or [] if absent (a phase
    config may legitimately leave some groups out)."""
    section = phase_cfg.get(group)
    if section is None:
        return []
    return list(section.get("prompts", []))


def _render_chat_prompt_ids(
    user_text: str,
    system: str | None,
    tokenizer: Tokenizer,
) -> list[int]:
    """Build the prompt token list for a single user turn, optionally
    behind a system message — mirrors what chat.py does for one turn."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return render_prompt_for_completion(messages, tokenizer)


def _generate_pretrain(
    text_gen: TextGenerator,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    if not prompts:
        return []
    return text_gen.generate(
        prompts,
        max_new_tokens=max_new_tokens,
        return_only_completion=True,
        include_special_in_output=False,
    )


def _generate_chat(
    *,
    model: LanguageModel,
    tokenizer: Tokenizer,
    generator: Generator,
    device: torch.device,
    prompts: list[str],
    system: str | None,
    max_new_tokens: int,
) -> list[str]:
    """One chat completion per prompt (system + user → assistant)."""
    completions: list[str] = []
    for prompt in prompts:
        ids = _render_chat_prompt_ids(prompt, system, tokenizer)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = generator.generate(model, input_ids, max_new_tokens=max_new_tokens)
        completion_ids = out[0].tolist()[len(ids):]
        completions.append(tokenizer.decode(completion_ids, include_special=False))
    return completions


def _emit_group_header(fh: IO[str], group: str, *, verbose: bool) -> None:
    header = f"--- GROUP: {GROUP_HEADERS[group]} ---"
    fh.write(f"\n{header}\n\n")
    if verbose:
        print(f"\n{header}\n")


def _emit_plain(fh: IO[str], prompt: str, completion: str, *, verbose: bool) -> None:
    fh.write(f"PROMPT:     {prompt}\n")
    fh.write(f"COMPLETION: {completion}\n\n")
    if verbose:
        print(f"PROMPT:     {prompt}")
        print(f"COMPLETION: {completion}\n")


def _emit_scored(
    fh: IO[str], prompt: str, ref: str, completion: str, score: float, *, verbose: bool
) -> None:
    pred = extract_final_answer(completion)
    fh.write(f"PROMPT:     {prompt}\n")
    fh.write(f"REFERENCE:  {ref}\n")
    fh.write(f"PRED:       {pred}\n")
    fh.write(f"SCORE:      {score:.1f}\n")
    fh.write(f"COMPLETION: {completion}\n\n")
    if verbose:
        print(f"PROMPT:     {prompt}")
        print(f"PRED: {pred} | REF: {ref} | SCORE: {score:.1f}")
        print(f"COMPLETION: {completion}\n")


def run_benchmark(
    *,
    model: LanguageModel,
    tokenizer: Tokenizer,
    generator: Generator,
    device: torch.device,
    phase: str,
    phase_cfg: dict[str, Any],
    output_path: Path,
    max_new_tokens: int,
    header_lines: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Execute the benchmark for one phase, write to `output_path`, and
    return per-group score lists for the rl phase ({} otherwise).

    `header_lines` is appended verbatim above the prompt sections so the
    caller can record what they want — checkpoint path + sampling config
    for the CLI, run step / model meta for the training callback. Don't
    include the `=== benchmark ===` banner here; we add it.

    The caller is responsible for putting `model` in eval mode and
    placing it on `device`; the generator must already be configured
    with the right `eos_id` for the phase (chat <|im_end|> for sft/rl,
    pretrain eos for pretrain).
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_gen = TextGenerator(model=model, tokenizer=tokenizer, generator=generator)
    system = phase_cfg.get("system")
    reward = GSM8KReward() if phase == "rl" else None
    rl_scores_by_group: dict[str, list[float]] = {}

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(f"=== benchmark: {phase} ===\n")
        for line in header_lines or []:
            fh.write(f"{line}\n")
        description = (phase_cfg.get("description") or "").strip()
        if description:
            fh.write(f"description:       {description}\n")
        fh.write("\n")

        # Fixed order near → generalize → ood. Makes diffs across runs /
        # checkpoints directly comparable and reads top-down from "what
        # should be perfect" to "what's outside scope".
        for group in GROUPS:
            items = _group_prompts(phase_cfg, group)
            if not items:
                continue
            _emit_group_header(fh, group, verbose=verbose)

            if phase == "pretrain":
                completions = _generate_pretrain(text_gen, list(items), max_new_tokens)
                for prompt, completion in zip(items, completions, strict=True):
                    _emit_plain(fh, prompt, completion, verbose=verbose)

            elif phase == "sft":
                completions = _generate_chat(
                    model=model, tokenizer=tokenizer, generator=generator,
                    device=device, prompts=list(items), system=system,
                    max_new_tokens=max_new_tokens,
                )
                for prompt, completion in zip(items, completions, strict=True):
                    _emit_plain(fh, prompt, completion, verbose=verbose)

            else:  # rl
                # `near` and `generalize` are scored ({prompt, reference}
                # dicts); `ood` is plain strings, no reference, no score.
                if group in ("near", "generalize"):
                    prompts_s = [it["prompt"] for it in items]
                    references = [str(it["reference"]) for it in items]
                    completions = _generate_chat(
                        model=model, tokenizer=tokenizer, generator=generator,
                        device=device, prompts=prompts_s, system=system,
                        max_new_tokens=max_new_tokens,
                    )
                    scores: list[float] = []
                    for prompt, ref, completion in zip(
                        prompts_s, references, completions, strict=True
                    ):
                        score = float(reward(completion, ref))  # type: ignore[misc]
                        scores.append(score)
                        _emit_scored(fh, prompt, ref, completion, score, verbose=verbose)
                    rl_scores_by_group[group] = scores
                else:
                    completions = _generate_chat(
                        model=model, tokenizer=tokenizer, generator=generator,
                        device=device, prompts=list(items), system=system,
                        max_new_tokens=max_new_tokens,
                    )
                    for prompt, completion in zip(items, completions, strict=True):
                        _emit_plain(fh, prompt, completion, verbose=verbose)

        if phase == "rl" and rl_scores_by_group:
            summary_lines = ["=== summary ==="]
            for group in GROUPS:
                if group not in rl_scores_by_group:
                    continue
                scores = rl_scores_by_group[group]
                rate = sum(scores) / len(scores) if scores else 0.0
                summary_lines.append(
                    f"  {group:<11} solve_rate {rate:.2%}  ({int(sum(scores))}/{len(scores)})"
                )
            summary = "\n".join(summary_lines) + "\n"
            fh.write(summary)
            if verbose:
                print(summary)

    return rl_scores_by_group
