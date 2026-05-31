"""Tests for the per-pipeline improvement summary (scripts/inference/aggregate_evals.py).

Pure data logic — exercised with hand-built RunEvals, no run dirs or torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# aggregate_evals lives under scripts/, which isn't an importable package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "inference"))

from aggregate_evals import (  # noqa: E402
    RunEvals,
    _lower_is_better,
    _scores,
    _stamped,
    build_improvements,
    build_pipelines,
    load_run,
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
        "pretrain_fineweb",
        "20260520_000000",
        "pretrain",
        {"arc_easy": {"acc": 0.40}, "lambada_openai": {"perplexity": 30.0, "acc": 0.30}},
    )
    dapt_arc = _run(
        "dapt_arc",
        "20260521_000000",
        "dapt",
        {"arc_easy": {"acc": 0.45}},
        targets={"arc_easy"},
    )
    dpo_arc = _run(
        "dpo_arc",
        "20260522_000000",
        "dpo",
        {"arc_easy": {"acc": 0.52}},
        targets={"arc_easy"},
    )
    dapt_lambada = _run(
        "dapt_lambada",
        "20260521_120000",
        "dapt",
        {"lambada_openai": {"perplexity": 21.0, "acc": 0.34}},
        targets={"lambada_openai"},
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
    assert arc.stages == [("pretrain", 0.40), ("dapt", 0.45), ("dpo", 0.52)]
    assert round(arc.delta, 2) == 0.12 and arc.improved is True

    # perplexity dropped 30 -> 21: a negative delta that counts as an improvement
    ppl = imps[("lambada_openai", "perplexity")]
    assert ppl.stages == [("pretrain", 30.0), ("dapt", 21.0)]
    assert ppl.delta < 0 and ppl.improved is True

    # lambada accuracy rose 0.30 -> 0.34: positive delta, also an improvement
    lacc = imps[("lambada_openai", "acc")]
    assert lacc.delta > 0 and lacc.improved is True


def test_shows_value_at_every_stage_including_sft_rl() -> None:
    # A pretrain -> sft -> rl pipeline must surface the value at *each* stage,
    # not just the endpoints, with a column per stage in the rendered table.
    runs = [
        _run("pretrain", "20260520_000000", "pretrain", {"piqa": {"acc": 0.60}}),
        _run("sft", "20260521_000000", "sft", {"piqa": {"acc": 0.66}}, targets={"piqa"}),
        _run("rl", "20260522_000000", "rl", {"piqa": {"acc": 0.71}}, targets={"piqa"}),
    ]
    (imp,) = build_improvements(runs)
    assert imp.stages == [("pretrain", 0.60), ("sft", 0.66), ("rl", 0.71)]
    assert round(imp.delta, 2) == 0.11 and imp.improved is True

    header = next(ln for ln in render_improvements_md(runs) if ln.startswith("| pipeline"))
    for stage in ("pretrain", "sft", "rl"):
        assert stage in header


def test_load_run_infers_stage_from_name_when_absent(tmp_path: Path) -> None:
    # A standalone eval JSON has no "stage"; it should be read off the run-name
    # prefix so the run isn't left unlabeled ("?").
    d = tmp_path / "20260520_000000_pretrain_fineweb"
    d.mkdir()
    (d / "lm_eval_x.json").write_text(
        json.dumps({"results": {"piqa": {"acc,none": 0.6}}}), encoding="utf-8"
    )
    (run,) = load_run(d)
    assert run.stage == "pretrain" and run.phase == "end"

    # An unrecognized prefix stays unlabeled rather than guessing.
    d2 = tmp_path / "20260520_000000_debug_shakespeare"
    d2.mkdir()
    (d2 / "lm_eval_x.json").write_text(
        json.dumps({"results": {"piqa": {"acc,none": 0.6}}}), encoding="utf-8"
    )
    (run2,) = load_run(d2)
    assert run2.stage is None


def test_load_run_splits_start_and_end_snapshots(tmp_path: Path) -> None:
    # A dir holding both an input (start) and result (end) snapshot yields two
    # RunEvals, named @start / @end, with the start ordered first.
    d = tmp_path / "20260522_000000_dpo_piqa"
    d.mkdir()
    (d / "lm_eval_target_start.json").write_text(
        json.dumps(
            {
                "stage": "dpo",
                "phase": "start",
                "split": "target",
                "results": {"piqa": {"acc,none": 0.56}},
            }
        ),
        encoding="utf-8",
    )
    (d / "lm_eval_target.json").write_text(
        json.dumps(
            {
                "stage": "dpo",
                "phase": "end",
                "split": "target",
                "results": {"piqa": {"acc,none": 0.67}},
            }
        ),
        encoding="utf-8",
    )
    runs = sorted(load_run(d), key=lambda r: r.phase)
    end, start = runs[0], runs[1]  # alphabetical: "end" < "start"
    assert start.phase == "start" and start.run_name == "dpo_piqa@start"
    assert end.phase == "end" and end.run_name == "dpo_piqa@end"
    assert start.metrics["piqa"]["acc"] == 0.56
    assert end.metrics["piqa"]["acc"] == 0.67
    assert start.target_tasks == {"piqa"} == end.target_tasks


def test_start_end_snapshot_makes_single_stage_branch_improve() -> None:
    # A DPO-only branch (no DAPT, no pretrain baseline in scope) still gets a
    # before->after row from its own start/end snapshots, in @start/@end columns.
    runs = [
        _run("dpo_piqa@start", "20260522_000000", "dpo", {"piqa": {"acc": 0.56}}, targets={"piqa"}),
        _run("dpo_piqa@end", "20260522_000000", "dpo", {"piqa": {"acc": 0.67}}, targets={"piqa"}),
    ]
    runs[0].phase, runs[1].phase = "start", "end"
    (imp,) = build_improvements(runs)
    assert imp.stages == [("dpo@start", 0.56), ("dpo@end", 0.67)]
    assert round(imp.delta, 2) == 0.11 and imp.improved is True

    header = next(ln for ln in render_improvements_md(runs) if ln.startswith("| pipeline"))
    assert "dpo@start" in header and "dpo@end" in header


def test_base_pinned_first_even_with_unknown_stage() -> None:
    # A baseline whose stage can't be resolved (stage="") must still lead the
    # chain, not sort to the end and invert the delta.
    runs = [
        _run("eval_dump", "20260520_000000", "", {"piqa": {"acc": 0.40}}),  # base, no stage
        _run("dpo_piqa", "20260521_000000", "dpo", {"piqa": {"acc": 0.55}}, targets={"piqa"}),
    ]
    (_task, chain) = build_pipelines(runs)[0]
    assert [r.run_name for r in chain] == ["eval_dump", "dpo_piqa"]  # base first
    (imp,) = build_improvements(runs)
    assert imp.stages == [("?", 0.40), ("dpo", 0.55)]
    assert imp.delta > 0 and imp.improved is True  # base -> dpo, not the inverse


def test_scores_drops_bookkeeping_fields() -> None:
    # sample_len / samples are numeric but not metrics; alias/stderr also dropped.
    block = {
        "alias": "piqa",
        "name": "piqa",
        "sample_len": 1838,
        "samples": 1838,
        "acc,none": 0.61,
        "acc_stderr,none": 0.012,
        "acc_norm,none": 0.60,
    }
    assert _scores(block) == {"acc": 0.61, "acc_norm": 0.60}


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
    # one column per stage present across pipelines
    header = next(ln for ln in render_improvements_md(_runs()) if ln.startswith("| pipeline"))
    for stage in ("pretrain", "dapt", "dpo"):
        assert stage in header
