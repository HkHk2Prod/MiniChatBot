"""Aggregate per-run lm-eval result JSONs into one side-by-side summary.

Each training stage writes its lm-eval scores into its own run dir:
`lm_eval_target.json` / `lm_eval_other.json` (the in-training callback) or
`lm_eval_<ts>.json` (the standalone scripts/inference/eval_harness.py). This
tool globs the run dirs, lines the runs up chronologically as columns, and
emits a single table so a branch's pretrain -> DAPT -> DPO progression reads
at a glance (target metrics should climb; collateral tasks may decay).

Examples:
    # everything under runs/, written to runs/eval_summary.md
    python scripts/inference/aggregate_evals.py

    # just one branch, also dump structured JSON
    python scripts/inference/aggregate_evals.py \\
        --filter pretrain_fineweb,dapt_arc,dpo_arc --json runs/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR_RE = re.compile(r"^(\d{8}_\d{6})_(.+)$")


@dataclass
class RunEvals:
    run_dir: str
    run_name: str
    timestamp: str  # YYYYMMDD_HHMMSS prefix, or "" if absent
    stage: str | None = None
    num_fewshot: int | None = None
    limit: int | None = None
    n_params: int | None = None
    # task -> metric -> value
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    target_tasks: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        return self.run_name


def _scores(task_results: dict) -> dict[str, float]:
    """Pull plain numeric metrics from an lm-eval task block.

    lm-eval keys look like ``"acc,none"`` / ``"acc_stderr,none"`` / ``"alias"``;
    keep the numeric, non-stderr ones and drop the ``,<filter>`` suffix.
    """
    out: dict[str, float] = {}
    for key, val in task_results.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        name = key.split(",")[0]
        if name == "alias" or name.endswith("_stderr"):
            continue
        out[name] = float(val)
    return out


def load_run(run_path: Path) -> RunEvals | None:
    files = sorted(run_path.glob("lm_eval*.json"))
    if not files:
        return None
    m = RUN_DIR_RE.match(run_path.name)
    timestamp, run_name = (m.group(1), m.group(2)) if m else ("", run_path.name)
    run = RunEvals(run_dir=run_path.name, run_name=run_name, timestamp=timestamp)
    for fpath in files:
        try:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[skip] {fpath}: {exc}")
            continue
        split = payload.get("split")  # "target" / "other" for the callback files
        run.stage = run.stage or payload.get("stage")
        run.num_fewshot = payload.get("num_fewshot", run.num_fewshot)
        run.limit = payload.get("limit", run.limit)
        run.n_params = payload.get("n_params", run.n_params)
        for task, task_results in (payload.get("results") or {}).items():
            scores = _scores(task_results)
            if not scores:
                continue
            run.metrics.setdefault(task, {}).update(scores)
            if split == "target":
                run.target_tasks.add(task)
    return run if run.metrics else None


def collect_runs(runs_dir: Path, name_filters: list[str]) -> list[RunEvals]:
    runs: list[RunEvals] = []
    for child in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run = load_run(child)
        if run is None:
            continue
        if name_filters and not any(f in run.run_name for f in name_filters):
            continue
        runs.append(run)
    runs.sort(key=lambda r: (r.timestamp, r.run_dir))
    return runs


def build_rows(runs: list[RunEvals]) -> list[tuple[str, str, bool]]:
    """Ordered (task, metric, is_target) rows; target tasks float to the top."""
    any_target = {t for r in runs for t in r.target_tasks}
    pairs: dict[str, set[str]] = {}
    for run in runs:
        for task, scores in run.metrics.items():
            pairs.setdefault(task, set()).update(scores)
    rows: list[tuple[str, str, bool]] = []
    for task in sorted(pairs, key=lambda t: (t not in any_target, t)):
        for metric in sorted(pairs[task]):
            rows.append((task, metric, task in any_target))
    return rows


def fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if abs(value) < 1 else f"{value:.3f}"


def render_markdown(runs: list[RunEvals], rows: list[tuple[str, str, bool]]) -> str:
    cols = [r.label for r in runs]
    lines = ["# lm-eval summary", ""]
    lines.append(f"_generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    # per-run metadata
    lines.append("| run | stage | num_fewshot | limit | params |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for r in runs:
        params = f"{r.n_params / 1e6:.1f}M" if r.n_params else "—"
        nfs = r.num_fewshot if r.num_fewshot is not None else "—"
        limit = r.limit if r.limit is not None else "full"
        lines.append(f"| {r.label} | {r.stage or '—'} | {nfs} | {limit} | {params} |")
    lines.append("")
    lines.append("★ = target task for some stage. Columns are chronological.")
    lines.append("")
    header = "| task | metric | " + " | ".join(cols) + " |"
    sep = "| --- | --- | " + " | ".join("---:" for _ in cols) + " |"
    lines += [header, sep]
    for task, metric, is_target in rows:
        name = f"★ {task}" if is_target else task
        cells = [fmt(r.metrics.get(task, {}).get(metric)) for r in runs]
        lines.append(f"| {name} | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_text(runs: list[RunEvals], rows: list[tuple[str, str, bool]]) -> str:
    cols = [r.label for r in runs]
    task_w = max([len("task")] + [len(("★ " if t else "") + tk) for tk, _, t in rows])
    metric_w = max([len("metric")] + [len(m) for _, m, _ in rows])
    col_w = [max(len(c), 8) for c in cols]
    out = []
    head = f"{'task':<{task_w}}  {'metric':<{metric_w}}  " + "  ".join(
        f"{c:>{w}}" for c, w in zip(cols, col_w, strict=True)
    )
    out.append(head)
    out.append("-" * len(head))
    for task, metric, is_target in rows:
        name = ("★ " if is_target else "") + task
        cells = "  ".join(
            f"{fmt(r.metrics.get(task, {}).get(metric)):>{w}}"
            for r, w in zip(runs, col_w, strict=True)
        )
        out.append(f"{name:<{task_w}}  {metric:<{metric_w}}  {cells}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--runs-dir", default="runs", help="Directory of run dirs (default: runs).")
    ap.add_argument(
        "--filter",
        default=None,
        help="Comma-separated run_name substrings to include (e.g. pretrain_fineweb,dapt_arc).",
    )
    ap.add_argument(
        "--output",
        default="runs/eval_summary.md",
        help="Markdown summary path (default: runs/eval_summary.md).",
    )
    ap.add_argument("--json", default=None, help="Also write structured JSON to this path.")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"runs dir not found: {runs_dir}")
    name_filters = [s.strip() for s in (args.filter or "").split(",") if s.strip()]

    runs = collect_runs(runs_dir, name_filters)
    if not runs:
        where = f" matching {name_filters}" if name_filters else ""
        raise SystemExit(f"No lm_eval*.json found under {runs_dir}/*{where}.")

    rows = build_rows(runs)
    print(render_text(runs, rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(runs, rows), encoding="utf-8")
    print(f"\nwrote {out_path}  ({len(runs)} runs, {len(rows)} task/metric rows)")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": [
                {
                    "run_dir": r.run_dir,
                    "run_name": r.run_name,
                    "timestamp": r.timestamp,
                    "stage": r.stage,
                    "num_fewshot": r.num_fewshot,
                    "limit": r.limit,
                    "n_params": r.n_params,
                    "target_tasks": sorted(r.target_tasks),
                    "metrics": r.metrics,
                }
                for r in runs
            ],
        }
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
