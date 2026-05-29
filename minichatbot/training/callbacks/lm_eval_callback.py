"""End-of-training lm-eval callback: writes target/other split result JSONs.

At `on_train_end`, runs EleutherAI's lm-evaluation-harness on the final
model and writes two files into the run dir:

    lm_eval_target.json  — tasks this stage is meant to *improve*
    lm_eval_other.json   — the rest (watch for collateral *degradation*)

Splitting them makes a branch's pretrain → DAPT → DPO progression easy to
read: the target metric should climb across stages while the collateral
tasks decay. Each payload mirrors `scripts/inference/eval_harness.py` so
the standalone script and this callback stay comparable.

Needs the optional `eval` extra (`pip install -e ".[eval]"`); like the
benchmark callback, any failure is logged and swallowed so it can't crash
an otherwise-successful run. Evaluates the in-memory (final-step) weights —
use `scripts/inference/eval_harness.py` to score a specific checkpoint.

Configure in a run's `callbacks:` block:

    - type: lm_eval
      params:
        target_tasks: [arc_easy]
        other_tasks: [piqa, hellaswag, lambada_openai, wikitext]
        num_fewshot: 0
        limit: null          # cap examples per task for a fast pass
        chat: false          # stop generation at <|im_end|> for chat ckpts
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
    ) -> None:
        self.target_tasks = list(target_tasks or [])
        self.other_tasks = list(other_tasks or [])
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.batch_size = batch_size
        self.max_gen_toks = max_gen_toks
        self.chat = chat

    def on_train_end(self, ctx: CallbackContext) -> None:
        try:
            self._run(ctx)
        except Exception as exc:  # noqa: BLE001 — never crash a finished run
            print(f"[lm_eval] skipped: {exc}")

    def _run(self, ctx: CallbackContext) -> None:
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

        for label, tasks in (("target", self.target_tasks), ("other", self.other_tasks)):
            if not tasks:
                continue
            print(
                f"[lm_eval] {label}: {tasks} (num_fewshot={self.num_fewshot}, limit={self.limit})"
            )
            results = evaluate_fn(
                model=adapter,
                tasks=list(tasks),
                num_fewshot=self.num_fewshot,
                limit=self.limit,
            )
            payload = {
                "stage": ctx.config.stage,
                "split": label,
                "tasks": list(tasks),
                "num_fewshot": self.num_fewshot,
                "limit": self.limit,
                "results": results.get("results", {}),
                "n_params": n_params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out = run_dir / f"lm_eval_{label}.json"
            with out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"[lm_eval] wrote {out}")
