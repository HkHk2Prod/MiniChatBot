"""Tests for the per-pipeline improvement summary (scripts/inference/aggregate_evals.py).

Pure data logic — exercised with hand-built RunEvals, no run dirs or torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

# aggregate_evals lives under scripts/, which isn't an importable package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "inference"))

from aggregate_evals import (  # noqa: E402
    RunEvals,
    _lower_is_better,
    _stamped,
    build_improvements,
    build_pipelines,
    render_improvements_md,
    render_markdown,
)


def _run(
    name: str, ts: str, stage: str, metrics: dict, targets: set[str] = frozenset()
) -> RunEvals:
    return RunEvals(
        run_dir=f"{ts}_{name}",
        run_name=name,
        timestamp=ts,
        stage=stage,
        metrics=metrics,
        target_tasks=set(targets),
    )


def _runs() -> list[RunEvals]:
    # A base pretrain plus two pipelines: arc (accuracy, higher=better) and
    # lambada (perplexity, lower=better). Stages forked from the same base.
    pretrain = _run(
        "pretrain_fineweb", "20260520_000000", "pretrain",
        {"arc_easy": {"acc": 0.40}, "lambada_openai": {"perplexity": 30.0, "acc": 0.30}},
    )
    dapt_arc = _run(
        "dapt_arc", "20260521_000000", "dapt",
        {"arc_easy": {"acc": 0.45}}, targets={"arc_easy"},
    )
    dpo_arc = _run(
        "dpo_arc", "20260522_000000", "dpo",
        {"arc_easy": {"acc": 0.52}}, targets={"arc_easy"},
    )
    dapt_lambada = _run(
        "dapt_lambada", "20260521_120000", "dapt",
        {"lambada_openai": {"perplexity": 21.0, "acc": 0.34}}, targets={"lambada_openai"},
    )
    # chronological order, as collect_runs would hand them over
    return [pretrain, dapt_arc, dpo_arc, dapt_lambada]


def test_stamped_prefixes_filename_like_run_dirs() -> None:
    # Timestamp goes in front of the name (matching <ts>_<run_name> dirs), not
    # before the extension, and the parent directory is preserved.
    assert _stamped(Path("runs/eval_summary.md"), "20260527_004147") == Path(
        "runs/20260527_004147_eval_summary.md"
    )
    assert _stamped(Path("out/dir/summary.json"), "20260101_000000") == Path(
        "out/dir/20260101_000000_summary.json"
    )


def test_lower_is_better_classifies_metric_families() -> None:
    for m in ("perplexity", "word_perplexity", "byte_perplexity", "bits_per_byte", "eval_loss"):
        assert _lower_is_better(m), m
    for m in ("acc", "acc_norm", "f1", "exact_match"):
        assert not _lower_is_better(m), m


def test_build_pipelines_groups_by_target_with_base_first() -> None:
    pipelines = {task: [r.run_name for r in chain] for task, chain in build_pipelines(_runs())}
    assert pipelines["arc_easy"] == ["pretrain_fineweb", "dapt_arc", "dpo_arc"]
    assert pipelines["lambada_openai"] == ["pretrain_fineweb", "dapt_lambada"]


def test_improvements_track_direction_per_metric() -> None:
    imps = {(i.task, i.metric): i for i in build_improvements(_runs())}

    arc = imps[("arc_easy", "acc")]
    assert arc.chain == "pretrain → dapt → dpo"
    assert arc.base == 0.40 and arc.final == 0.52
    assert round(arc.delta, 2) == 0.12 and arc.improved is True

    # perplexity dropped 30 -> 21: a negative delta that counts as an improvement
    ppl = imps[("lambada_openai", "perplexity")]
    assert ppl.base == 30.0 and ppl.final == 21.0
    assert ppl.delta < 0 and ppl.improved is True

    # lambada accuracy rose 0.30 -> 0.34: positive delta, also an improvement
    lacc = imps[("lambada_openai", "acc")]
    assert lacc.delta > 0 and lacc.improved is True


def test_regression_is_flagged_not_improved() -> None:
    runs = [
        _run("pretrain", "20260520_000000", "pretrain", {"piqa": {"acc": 0.70}}),
        _run("dpo_piqa", "20260521_000000", "dpo", {"piqa": {"acc": 0.66}}, targets={"piqa"}),
    ]
    (imp,) = build_improvements(runs)
    assert imp.delta < 0 and imp.improved is False


def test_metric_at_single_stage_is_skipped() -> None:
    # arc_easy only evaluated at the dpo stage -> no start/final pair -> no row.
    runs = [
        _run("pretrain", "20260520_000000", "pretrain", {}),
        _run("dpo_arc", "20260521_000000", "dpo", {"arc_easy": {"acc": 0.5}}, targets={"arc_easy"}),
    ]
    assert build_improvements(runs) == []


def test_no_targets_yields_no_pipeline_section() -> None:
    # Standalone evals carry no stage/target; the section is omitted entirely.
    runs = [_run("adhoc", "20260520_000000", "", {"wikitext": {"word_perplexity": 40.0}})]
    assert build_improvements(runs) == []
    assert render_improvements_md(runs) == []
    rows = []  # body rows are irrelevant here
    assert "pipeline improvements" not in render_markdown(runs, rows)


def test_markdown_includes_improvement_table() -> None:
    md = render_markdown(_runs(), rows=[])
    assert "## pipeline improvements" in md
    assert "## all scores" in md
    assert "✓" in md  # at least one pipeline improved
    assert "pretrain → dapt → dpo" in md
