"""lm-eval callback: writes target/other split result JSONs at start and/or end.

Runs EleutherAI's lm-evaluation-harness on the model and writes two files
into the run dir per phase:

    lm_eval_target.json        — tasks this stage is meant to *improve* (end)
    lm_eval_other.json         — the rest (watch for collateral *degradation*)
    lm_eval_target_start.json  — same, scored at `on_train_start` (the *input*
    lm_eval_other_start.json     checkpoint, before any optimizer step)

Splitting target/other makes a branch's pretrain → DAPT → DPO progression
easy to read: the target metric should climb across stages while the
collateral tasks decay. The `_start` snapshot lets a single stage record its
own before→after — useful when a stage's input baseline isn't otherwise on
record (e.g. a DPO branch with no DAPT ahead of it scores its pretrain input
at start instead of borrowing the previous stage's end numbers). Each payload
mirrors `scripts/inference/eval_harness.py` so the standalone script and this
callback stay comparable.

Needs the optional `eval` extra (`pip install -e ".[eval]"`); like the
benchmark callback, any failure is logged and swallowed so it can't crash
an otherwise-successful run. Evaluates the in-memory weights (input at start,
final-step at end) — use `scripts/inference/eval_harness.py` to score a
specific checkpoint.

Configure in a run's `callbacks:` block:

    - type: lm_eval
      params:
        target_tasks: [arc_easy]
        other_tasks: [piqa, hellaswag, lambada_openai, wikitext]
        num_fewshot: 0
        limit: null          # cap examples per task for a fast pass
        chat: false          # stop generation at <|im_end|> for chat ckpts
        eval_at: end         # "end" (default) | "start" | "both"
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minichatbot.tokenizer.bpe import IM_END_TOKEN
from minichatbot.training.callbacks import CALLBACK_REGISTRY
from minichatbot.training.callbacks.base import Callback, CallbackContext
from minichatbot.utils.torch_helpers import unwrap_compiled

_VALID_EVAL_AT = ("end", "start", "both")


@CALLBACK_REGISTRY.register("lm_eval")
class LmEvalCallback(Callback):
    def __init__(
        self,
        target_tasks: list[str] | None = None,
        other_tasks: list[str] | None = None,
        num_fewshot: int = 0,
        limit: int | None = None,
        batch_size: int = 8,
        max_gen_toks: int = 256,
        chat: bool = False,
        eval_at: str = "end",
    ) -> None:
        if eval_at not in _VALID_EVAL_AT:
            raise ValueError(f"eval_at must be one of {_VALID_EVAL_AT}, got {eval_at!r}")
        self.target_tasks = list(target_tasks or [])
        self.other_tasks = list(other_tasks or [])
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.batch_size = batch_size
        self.max_gen_toks = max_gen_toks
        self.chat = chat
        self.eval_at = eval_at

    def on_train_start(self, ctx: CallbackContext) -> None:
        # Scores the input checkpoint (loaded before the first optimizer step),
        # so this stage records its own baseline rather than relying on the
        # previous stage's end-of-training numbers.
        if self.eval_at not in ("start", "both"):
            return
        try:
            self._run(ctx, phase="start")
        except Exception as exc:  # noqa: BLE001 — never crash a starting run
            print(f"[lm_eval] start eval skipped: {exc}")

    def on_train_end(self, ctx: CallbackContext) -> None:
        if self.eval_at not in ("end", "both"):
            return
        try:
            self._run(ctx, phase="end")
        except Exception as exc:  # noqa: BLE001 — never crash a finished run
            print(f"[lm_eval] skipped: {exc}")

    def _run(self, ctx: CallbackContext, phase: str = "end") -> None:
        if not self.target_tasks and not self.other_tasks:
            print("[lm_eval] skipped: no target_tasks or other_tasks configured.")
            return
        if ctx.tokenizer is None:
            print("[lm_eval] skipped: trainer didn't provide a tokenizer.")
            return
        try:
            import lm_eval
        except ModuleNotFoundError:
            print('[lm_eval] skipped: install the eval extra (pip install -e ".[eval]").')
            return

        from minichatbot.eval.lm_eval_adapter import MiniChatBotLM

        model = unwrap_compiled(ctx.model).eval()
        device = next(model.parameters()).device
        tokenizer = ctx.tokenizer
        eos_id = tokenizer.special_token_id(IM_END_TOKEN) if self.chat else None
        adapter = MiniChatBotLM(
            model,
            tokenizer,
            device=device,
            batch_size=self.batch_size,
            max_gen_toks=self.max_gen_toks,
            eos_id=eos_id,
        )
        n_params = sum(p.numel() for p in model.parameters())
        evaluate_fn: Any = lm_eval.simple_evaluate
        run_dir = Path(ctx.run_dir)
        # Start-phase snapshots get a "_start" suffix so they sit alongside the
        # end-phase files in the same run dir without overwriting them.
        suffix = "_start" if phase == "start" else ""

        for label, tasks in (("target", self.target_tasks), ("other", self.other_tasks)):
            if not tasks:
                continue
            print(
                f"[lm_eval] {phase} {label}: {tasks} "
                f"(num_fewshot={self.num_fewshot}, limit={self.limit})"
            )
            results = evaluate_fn(
                model=adapter,
                tasks=list(tasks),
                num_fewshot=self.num_fewshot,
                limit=self.limit,
            )
            payload = {
                "stage": ctx.config.stage,
                "phase": phase,
                "split": label,
                "tasks": list(tasks),
                "num_fewshot": self.num_fewshot,
                "limit": self.limit,
                "results": results.get("results", {}),
                "n_params": n_params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out = run_dir / f"lm_eval_{label}{suffix}.json"
            with out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"[lm_eval] wrote {out}")
