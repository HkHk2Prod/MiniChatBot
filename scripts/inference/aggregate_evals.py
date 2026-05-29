"""Aggregate per-run lm-eval result JSONs into one side-by-side summary.

Each training stage writes its lm-eval scores into its own run dir:
`lm_eval_target.json` / `lm_eval_other.json` (the in-training callback) or
`lm_eval_<ts>.json` (the standalone scripts/inference/eval_harness.py). This
tool globs the run dirs, lines the runs up chronologically as columns, and
emits a single table so a branch's pretrain -> DAPT -> DPO progression reads
at a glance (target metrics should climb; collateral tasks may decay).

The output filename is timestamp-prefixed like run dirs, so successive
aggregations archive rather than overwrite: runs/<YYYYMMDD_HHMMSS>_eval_summary.md.

Examples:
    # everything under runs/, written to runs/<timestamp>_eval_summary.md
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


# Numeric fields lm-eval/eval_harness emit alongside real metrics that aren't
# performance signals (constant per task), so they're dropped from the tables.
_NON_METRIC_KEYS = frozenset({"sample_len", "samples"})


def _scores(task_results: dict) -> dict[str, float]:
    """Pull plain numeric metrics from an lm-eval task block.

    lm-eval keys look like ``"acc,none"`` / ``"acc_stderr,none"`` / ``"alias"``;
    keep the numeric, non-stderr ones and drop the ``,<filter>`` suffix. Bookkeeping
    fields like ``sample_len`` are numeric but not metrics, so they're skipped too.
    """
    out: dict[str, float] = {}
    for key, val in task_results.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        name = key.split(",")[0]
        if name == "alias" or name.endswith("_stderr") or name in _NON_METRIC_KEYS:
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
    # Standalone evals (eval_harness.py) write no "stage"; fall back to the
    # run-name prefix (pretrain_fineweb -> pretrain) when it names a known stage,
    # so such runs still land in the right pipeline column instead of "?".
    if run.stage is None:
        prefix = run.run_name.split("_", 1)[0].lower()
        if prefix in _STAGE_RANK:
            run.stage = prefix
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


def fmt_delta(value: float) -> str:
    """Signed change, matching fmt's precision switch."""
    return f"{value:+.4f}" if abs(value) < 1 else f"{value:+.3f}"


def _stamped(path: Path, stamp: str) -> Path:
    """Prefix the filename with a timestamp, matching run-dir naming
    (``<stamp>_<name>``): ``runs/eval_summary.md`` -> ``runs/<stamp>_eval_summary.md``."""
    return path.with_name(f"{stamp}_{path.name}")


# Canonical pipeline order; stages not listed sort after these, by name.
_STAGE_RANK = {"pretrain": 0, "dapt": 1, "sft": 2, "dpo": 3, "rl": 4}
# Metric families where a *lower* number is the improvement (perplexity / loss).
_LOWER_IS_BETTER = ("perplexity", "loss", "bits_per_byte")


def _stage_rank(stage: str | None) -> int:
    return _STAGE_RANK.get((stage or "").lower(), 50)


def _lower_is_better(metric: str) -> bool:
    m = metric.lower()
    return any(hint in m for hint in _LOWER_IS_BETTER)


@dataclass
class Improvement:
    task: str  # the pipeline's target task
    metric: str
    stages: list[tuple[str, float]]  # (stage, value) at each stage, in pipeline order
    delta: float  # value at the last stage minus the first
    improved: bool  # moved in the better direction for this metric


def _stage_columns(imps: list[Improvement]) -> list[str]:
    """Union of stage names across improvements, in canonical pipeline order."""
    seen = {stage for i in imps for stage, _ in i.stages}
    return sorted(seen, key=lambda s: (_stage_rank(s), s))


def build_pipelines(runs: list[RunEvals]) -> list[tuple[str, list[RunEvals]]]:
    """Group runs into per-target-task pipelines: the base pretrain followed by
    every stage that targeted that task, ordered along the pipeline.

    A stage declares its target task(s) through the in-training lm-eval callback
    (the ``split="target"`` file); the base is the pretrain those stages forked
    from. Standalone evals carry no stage/target, so they form no pipelines.

    The base leads the chain unconditionally — a base whose stage can't be
    resolved (sorts last by rank) must not be mistaken for the final stage,
    which would invert every delta. Only the targeted stages are rank-sorted.
    Returns ``[(task, [base, ...stages]), ...]`` sorted by task name.
    """
    base = next((r for r in runs if (r.stage or "").lower() == "pretrain"), None)
    if base is None:
        base = next((r for r in runs if not r.target_tasks), None)
    targets = sorted({t for r in runs for t in r.target_tasks})
    pipelines: list[tuple[str, list[RunEvals]]] = []
    for task in targets:
        stages = [r for r in runs if task in r.target_tasks]
        stages.sort(key=lambda r: (_stage_rank(r.stage), r.timestamp, r.run_dir))
        chain = [base, *stages] if base is not None and base not in stages else stages
        pipelines.append((task, chain))
    return pipelines


def build_improvements(runs: list[RunEvals]) -> list[Improvement]:
    """One row per (pipeline target task, metric), carrying the metric's value at
    every stage of the pipeline so progress shows step by step. Metrics seen at
    fewer than two stages are skipped — a single data point has no change."""
    out: list[Improvement] = []
    for task, chain in build_pipelines(runs):
        for metric in sorted({m for r in chain for m in r.metrics.get(task, {})}):
            stages = [
                (r.stage or "?", r.metrics[task][metric])
                for r in chain
                if metric in r.metrics.get(task, {})
            ]
            if len(stages) < 2:
                continue
            delta = stages[-1][1] - stages[0][1]
            improved = delta < 0 if _lower_is_better(metric) else delta > 0
            out.append(Improvement(task, metric, stages, delta, improved))
    return out


def _md_table(headers: list[str], aligns: list[str], rows: list[list[str]]) -> list[str]:
    """Render a GitHub-flavored Markdown table with cells padded to column width.

    Renderers ignore the extra padding, so the table looks identical once
    rendered — but the raw file stays column-aligned when read as plain text.
    ``aligns`` is per-column "l" (left) or "r" (right); the separator row gets
    a trailing ``:`` for right-aligned columns.
    """
    widths = [max(3, len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def cell(text: str, w: int, align: str) -> str:
        return text.rjust(w) if align == "r" else text.ljust(w)

    def sep(w: int, align: str) -> str:
        return "-" * (w - 1) + ":" if align == "r" else "-" * w

    out = [
        "| "
        + " | ".join(cell(h, w, a) for h, w, a in zip(headers, widths, aligns, strict=True))
        + " |"
    ]
    out.append("| " + " | ".join(sep(w, a) for w, a in zip(widths, aligns, strict=True)) + " |")
    for row in rows:
        cells = " | ".join(cell(c, w, a) for c, w, a in zip(row, widths, aligns, strict=True))
        out.append(f"| {cells} |")
    return out


def render_improvements_md(runs: list[RunEvals]) -> list[str]:
    """The small per-pipeline improvement table (empty list if no pipelines)."""
    imps = build_improvements(runs)
    if not imps:
        return []
    stage_cols = _stage_columns(imps)
    lines = [
        "## pipeline improvements",
        "",
        "_Target-task metric at each pipeline stage, with the net change. "
        "✓ = moved the right way: higher accuracy, or lower perplexity._",
        "",
    ]
    rows = []
    for i in imps:
        by_stage = dict(i.stages)
        rows.append(
            [
                i.task,
                i.metric,
                *[fmt(by_stage.get(s)) for s in stage_cols],
                fmt_delta(i.delta),
                "✓" if i.improved else "✗",
            ]
        )
    lines += _md_table(
        ["pipeline", "metric", *stage_cols, "Δ", "ok"],
        ["l", "l", *["r"] * len(stage_cols), "r", "l"],
        rows,
    )
    return lines


def render_markdown(runs: list[RunEvals], rows: list[tuple[str, str, bool]]) -> str:
    cols = [r.label for r in runs]
    lines = ["# lm-eval summary", ""]
    lines.append(f"_generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    # per-run metadata
    meta_rows = []
    for r in runs:
        params = f"{r.n_params / 1e6:.1f}M" if r.n_params else "—"
        nfs = str(r.num_fewshot) if r.num_fewshot is not None else "—"
        limit = str(r.limit) if r.limit is not None else "full"
        meta_rows.append([r.label, r.stage or "—", nfs, limit, params])
    lines += _md_table(
        ["run", "stage", "num_fewshot", "limit", "params"],
        ["l", "l", "r", "r", "r"],
        meta_rows,
    )
    lines.append("")
    # small per-pipeline summary, above the full all-runs table
    imp_lines = render_improvements_md(runs)
    if imp_lines:
        lines += imp_lines
        lines.append("")
    lines.append("## all scores")
    lines.append("")
    lines.append("★ = target task for some stage. Columns are chronological.")
    lines.append("")
    body_rows = [
        [f"★ {task}" if is_target else task, metric]
        + [fmt(r.metrics.get(task, {}).get(metric)) for r in runs]
        for task, metric, is_target in rows
    ]
    lines += _md_table(
        ["task", "metric", *cols],
        ["l", "l", *["r"] * len(cols)],
        body_rows,
    )
    return "\n".join(lines) + "\n"


def render_improvements_text(runs: list[RunEvals]) -> str:
    """Console version of the per-pipeline improvement table ("" if none)."""
    imps = build_improvements(runs)
    if not imps:
        return ""
    stage_cols = _stage_columns(imps)
    task_w = max([len("pipeline"), *(len(i.task) for i in imps)])
    metric_w = max([len("metric"), *(len(i.metric) for i in imps)])
    col_w = {s: max(len(s), 9) for s in stage_cols}
    head = (
        f"{'pipeline':<{task_w}}  {'metric':<{metric_w}}  "
        + "  ".join(f"{s:>{col_w[s]}}" for s in stage_cols)
        + f"  {'delta':>9}  ok"
    )
    out = ["pipeline improvements (target metric at each stage; net delta):", head, "-" * len(head)]
    for i in imps:
        by_stage = dict(i.stages)
        cells = "  ".join(f"{fmt(by_stage.get(s)):>{col_w[s]}}" for s in stage_cols)
        out.append(
            f"{i.task:<{task_w}}  {i.metric:<{metric_w}}  {cells}  "
            f"{fmt_delta(i.delta):>9}  {'yes' if i.improved else 'no'}"
        )
    return "\n".join(out)


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
        help="Markdown summary path; a <timestamp>_ prefix is added to the "
        "filename (default: runs/eval_summary.md -> runs/<timestamp>_eval_summary.md).",
    )
    ap.add_argument(
        "--json",
        default=None,
        help="Also write structured JSON here (same <timestamp>_ filename prefix).",
    )
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
    imp_text = render_improvements_text(runs)
    if imp_text:
        print(imp_text)
        print()
    print(render_text(runs, rows))

    # Timestamp the output filename as a prefix, like run dirs (builders.py).
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _stamped(Path(args.output), stamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(runs, rows), encoding="utf-8")
    print(f"\nwrote {out_path}  ({len(runs)} runs, {len(rows)} task/metric rows)")

    if args.json:
        json_path = _stamped(Path(args.json), stamp)
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
            "pipelines": [
                {
                    "task": i.task,
                    "metric": i.metric,
                    "stages": [{"stage": s, "value": v} for s, v in i.stages],
                    "delta": i.delta,
                    "improved": i.improved,
                }
                for i in build_improvements(runs)
            ],
        }
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
