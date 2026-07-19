from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .metrics import (
    ConvergenceSummary,
    MetricPoint,
    confirmation_decision,
    promotion_decision,
    rank_promoted_candidates,
    summarize_convergence,
)
from .models import RunSpec, RunState, StudySpec, StudyState
from .runtime import StateStore, atomic_write_json, cleanup_run_weights, reconcile_state, study_directory, utc_now


def load_metric_points(run_dir: Path, accuracy_key: str = "probe/validation_accuracy") -> tuple[MetricPoint, ...]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return ()
    points: list[MetricPoint] = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        accuracy = record.get(accuracy_key)
        active_seconds = record.get("convergence/active_seconds")
        step = record.get("_step")
        if accuracy is not None and active_seconds is not None and step is not None:
            points.append(MetricPoint(int(step), float(active_seconds), float(accuracy)))
    return tuple(points)


def _completed_pretrain_points(state: StudyState) -> dict[str, tuple[MetricPoint, ...]]:
    return {
        run_id: points
        for run_id, run in state.runs.items()
        if run.spec.kind == "pretrain"
        and run.status == "completed"
        and run.run_dir is not None
        and (points := load_metric_points(Path(run.run_dir)))
    }


def _last_metric(run_dir: Path, key: str) -> float | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    values = [json.loads(line).get(key) for line in metrics_path.read_text().splitlines() if line.strip()]
    present_values = [float(value) for value in values if value is not None]
    return present_values[-1] if present_values else None


def calculate_study_summaries(
    state: StudyState,
    spec: StudySpec,
) -> dict[str, ConvergenceSummary]:
    points_by_run = _completed_pretrain_points(state)
    baseline_id = f"pretrain-{spec.baseline.id}-seed0"
    baseline_points = points_by_run.get(baseline_id)
    if baseline_points is None:
        return {}
    baseline_peak = max(point.accuracy for point in baseline_points)
    common_step_horizon = min(points[-1].step for points in points_by_run.values())
    common_time_horizon = min(points[-1].active_seconds for points in points_by_run.values())
    return {
        run_id: summarize_convergence(
            points,
            baseline_peak,
            step_horizon=common_step_horizon,
            active_time_horizon=common_time_horizon,
        )
        for run_id, points in points_by_run.items()
    }


def calculate_sft_summaries(state: StudyState, spec: StudySpec) -> dict[str, Any]:
    completed = [
        run
        for run in state.runs.values()
        if run.spec.kind == "sft" and run.status == "completed" and run.run_dir is not None
    ]
    run_summaries: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    for shots in spec.evaluation.shots_per_class:
        shot_runs = [run for run in completed if run.spec.shots_per_class == shots]
        points_by_run = {
            run.spec.id: points
            for run in shot_runs
            if run.run_dir is not None
            and (points := load_metric_points(Path(run.run_dir), accuracy_key="sft/validation_accuracy"))
        }
        baseline_seed_zero_id = f"sft-{spec.baseline.id}-{'full' if shots is None else f'{shots}shot'}-seed0"
        baseline_points = points_by_run.get(baseline_seed_zero_id)
        if baseline_points is None:
            continue
        baseline_peak = max(point.accuracy for point in baseline_points)
        common_step_horizon = min(points[-1].step for points in points_by_run.values())
        common_time_horizon = min(points[-1].active_seconds for points in points_by_run.values())
        for run in shot_runs:
            points = points_by_run.get(run.spec.id)
            if points is None or run.run_dir is None:
                continue
            convergence = summarize_convergence(
                points,
                baseline_peak,
                step_horizon=common_step_horizon,
                active_time_horizon=common_time_horizon,
            )
            run_summaries[run.spec.id] = {
                **convergence.to_dict(),
                "test_accuracy": _last_metric(Path(run.run_dir), "sft/test_accuracy"),
            }
        shot_name = "full" if shots is None else f"{shots}shot"
        for variant in (spec.baseline.id, state.winner):
            if variant is None:
                continue
            variant_run_ids = [f"sft-{variant}-{shot_name}-seed{seed}" for seed in spec.evaluation.seeds]
            if not all(run_id in run_summaries for run_id in variant_run_ids):
                continue
            peak_values = [float(run_summaries[run_id]["peak_accuracy"]) for run_id in variant_run_ids]
            test_values = [run_summaries[run_id]["test_accuracy"] for run_id in variant_run_ids]
            if any(value is None for value in test_values):
                continue
            present_test_values = [float(value) for value in test_values if value is not None]
            aggregates[f"{variant}/{shot_name}"] = {
                "validation_peak_mean": statistics.mean(peak_values),
                "validation_peak_std": statistics.stdev(peak_values),
                "test_mean": statistics.mean(present_test_values),
                "test_std": statistics.stdev(present_test_values),
            }
        baseline_key = f"{spec.baseline.id}/{shot_name}"
        winner_key = f"{state.winner}/{shot_name}"
        if baseline_key in aggregates and winner_key in aggregates:
            baseline_test = [
                float(run_summaries[f"sft-{spec.baseline.id}-{shot_name}-seed{seed}"]["test_accuracy"])
                for seed in spec.evaluation.seeds
            ]
            winner_test = [
                float(run_summaries[f"sft-{state.winner}-{shot_name}-seed{seed}"]["test_accuracy"])
                for seed in spec.evaluation.seeds
            ]
            aggregates[winner_key]["paired_test_differences"] = [
                winner - baseline for baseline, winner in zip(baseline_test, winner_test, strict=True)
            ]
    return {"runs": run_summaries, "aggregates": aggregates}


def calculate_pretraining_aggregates(
    state: StudyState,
    spec: StudySpec,
    summaries: Mapping[str, ConvergenceSummary],
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    variants = (spec.baseline.id, state.winner)
    summaries_by_variant: dict[str, list[ConvergenceSummary]] = {}
    for variant in variants:
        if variant is None:
            continue
        run_ids = [f"pretrain-{variant}-seed{seed}" for seed in spec.seeds[:3]]
        if not all(run_id in summaries for run_id in run_ids):
            continue
        variant_summaries = [summaries[run_id] for run_id in run_ids]
        summaries_by_variant[variant] = variant_summaries
        peaks = [summary.peak_accuracy for summary in variant_summaries]
        aucs = [summary.active_time_auc for summary in variant_summaries]
        times = [summary.active_seconds_to_95 for summary in variant_summaries]
        present_times = [value for value in times if value is not None]
        all_reached_95 = len(present_times) == len(times)
        aggregates[variant] = {
            "peak_accuracy_mean": statistics.mean(peaks),
            "peak_accuracy_std": statistics.stdev(peaks),
            "active_time_auc_mean": statistics.mean(aucs),
            "active_time_auc_std": statistics.stdev(aucs),
            "active_seconds_to_95_mean": (statistics.mean(present_times) if all_reached_95 else None),
            "active_seconds_to_95_std": (statistics.stdev(present_times) if all_reached_95 else None),
            "censored_95_count": sum(value is None for value in times),
        }
    if state.winner is not None and spec.baseline.id in summaries_by_variant and state.winner in summaries_by_variant:
        baseline_summaries = summaries_by_variant[spec.baseline.id]
        winner_summaries = summaries_by_variant[state.winner]
        aggregates[state.winner]["paired_peak_differences"] = [
            winner.peak_accuracy - baseline.peak_accuracy
            for baseline, winner in zip(baseline_summaries, winner_summaries, strict=True)
        ]
        aggregates[state.winner]["paired_active_time_auc_differences"] = [
            winner.active_time_auc - baseline.active_time_auc
            for baseline, winner in zip(baseline_summaries, winner_summaries, strict=True)
        ]
        seed_zero = promotion_decision(baseline_summaries[0], winner_summaries[0], spec.promotion)
        if seed_zero.criterion is not None:
            aggregates["confirmation"] = asdict(
                confirmation_decision(
                    baseline_summaries,
                    winner_summaries,
                    seed_zero.criterion,
                    spec.promotion,
                )
            )
    return aggregates


def _add_replication_runs(state: StudyState, spec: StudySpec, winner: str) -> None:
    variant_by_id = {variant.id: variant for variant in (spec.baseline, *spec.variants)}
    for variant_id, role in ((spec.baseline.id, "baseline"), (winner, "winner")):
        variant = variant_by_id[variant_id]
        for seed in spec.seeds[1:3]:
            run_spec = RunSpec(
                id=f"pretrain-{variant_id}-seed{seed}",
                kind="pretrain",
                variant=variant_id,
                config=variant.config,
                seed=seed,
                role=role,
                evaluate_test=False,
            )
            state.runs.setdefault(run_spec.id, RunState(run_spec))
    if sum(run.spec.kind == "pretrain" for run in state.runs.values()) > spec.resources.max_pretraining_trials:
        raise RuntimeError("replication schedule exceeds the eight-trial pretraining limit")


def _add_exploration_runs(state: StudyState, spec: StudySpec) -> None:
    already_scheduled = {run.spec.variant for run in state.runs.values() if run.spec.kind == "pretrain"}
    remaining_slots = spec.resources.max_pretraining_trials - sum(
        run.spec.kind == "pretrain" for run in state.runs.values()
    )
    for variant in (variant for variant in spec.variants if variant.id not in already_scheduled):
        if remaining_slots <= 0:
            break
        run_spec = RunSpec(
            id=f"pretrain-{variant.id}-seed0",
            kind="pretrain",
            variant=variant.id,
            config=variant.config,
            seed=0,
            role="exploratory",
        )
        state.runs[run_spec.id] = RunState(run_spec)
        remaining_slots -= 1


def _add_sft_runs(state: StudyState, spec: StudySpec, winner: str) -> None:
    if spec.evaluation.finetune_config is None:
        return
    for variant, role in ((spec.baseline.id, "baseline"), (winner, "winner")):
        for seed in spec.evaluation.seeds:
            source_run = state.runs[f"pretrain-{variant}-seed{seed}"]
            if source_run.run_dir is None:
                raise RuntimeError(f"pretraining run directory is unavailable for {source_run.spec.id}")
            source_checkpoint = Path(source_run.run_dir) / "backbone.safetensors"
            for shots in spec.evaluation.shots_per_class:
                shot_name = "full" if shots is None else f"{shots}shot"
                run_spec = RunSpec(
                    id=f"sft-{variant}-{shot_name}-seed{seed}",
                    kind="sft",
                    variant=variant,
                    config=spec.evaluation.finetune_config,
                    seed=seed,
                    role=role,
                    source_checkpoint=source_checkpoint,
                    shots_per_class=shots,
                    subset_seed=seed,
                    evaluate_test=True,
                )
                state.runs.setdefault(run_spec.id, RunState(run_spec))


def advance_study(state: StudyState, spec: StudySpec, summaries: Mapping[str, ConvergenceSummary]) -> None:
    if state.phase in ("screening", "exploration"):
        phase_runs = [run for run in state.runs.values() if run.spec.kind == "pretrain" and run.spec.seed == 0]
        if not phase_runs or any(run.status not in ("completed", "failed", "timed_out") for run in phase_runs):
            return
        baseline_id = f"pretrain-{spec.baseline.id}-seed0"
        baseline_summary = summaries.get(baseline_id)
        if baseline_summary is None:
            return
        qualifying: list[tuple[str, ConvergenceSummary]] = []
        for run in phase_runs:
            if run.spec.variant == spec.baseline.id:
                run.decision = "baseline"
                continue
            candidate_summary = summaries.get(run.spec.id)
            if candidate_summary is None:
                continue
            decision = promotion_decision(baseline_summary, candidate_summary, spec.promotion)
            run.decision = "promoted" if decision.promoted else "rejected"
            if decision.promoted:
                qualifying.append((run.spec.variant, candidate_summary))
        if qualifying:
            winner = rank_promoted_candidates(qualifying)[0][0]
            state.winner = winner
            _add_replication_runs(state, spec, winner)
            state.phase = "confirmation"
        elif state.phase == "screening":
            before = len(state.runs)
            _add_exploration_runs(state, spec)
            state.phase = "exploration" if len(state.runs) > before else "no-promotion"
        else:
            state.phase = "no-promotion"

    if state.phase == "confirmation" and state.winner is not None:
        baseline_runs = [state.runs.get(f"pretrain-{spec.baseline.id}-seed{seed}") for seed in spec.seeds[:3]]
        candidate_runs = [state.runs.get(f"pretrain-{state.winner}-seed{seed}") for seed in spec.seeds[:3]]
        paired_runs = (*baseline_runs, *candidate_runs)
        if any(run is None or run.status != "completed" for run in paired_runs):
            return
        baseline_summaries = [summaries[run.spec.id] for run in baseline_runs if run is not None]
        candidate_summaries = [summaries[run.spec.id] for run in candidate_runs if run is not None]
        seed_zero_decision = promotion_decision(baseline_summaries[0], candidate_summaries[0], spec.promotion)
        if seed_zero_decision.criterion is None:
            state.phase = "not-confirmed"
            return
        confirmation = confirmation_decision(
            baseline_summaries,
            candidate_summaries,
            seed_zero_decision.criterion,
            spec.promotion,
        )
        if confirmation.confirmed:
            for run in candidate_runs:
                assert run is not None
                run.decision = "confirmed"
            state.phase = "evaluation"
            _add_sft_runs(state, spec, state.winner)
            if not any(run.spec.kind == "sft" for run in state.runs.values()):
                state.phase = "complete"
        else:
            for run in candidate_runs:
                assert run is not None
                run.decision = "rejected"
            state.phase = "not-confirmed"

    if state.phase == "evaluation":
        evaluation_runs = [run for run in state.runs.values() if run.spec.kind == "sft"]
        if evaluation_runs and all(run.status in ("completed", "failed", "timed_out") for run in evaluation_runs):
            state.phase = "complete"


def summarize_study(spec: StudySpec, spec_path: Path, repo_root: Path) -> dict[str, Any]:
    study_dir = study_directory(spec, repo_root)
    with StateStore(study_dir) as store:
        state = store.load_or_create(spec, spec_path)
        reconcile_state(state)
        summaries = calculate_study_summaries(state, spec)
        sft_summaries = calculate_sft_summaries(state, spec)
        pretraining_aggregates = calculate_pretraining_aggregates(state, spec, summaries)
        advance_study(state, spec, summaries)
        store.save(state)
        payload = {
            "study_id": spec.id,
            "phase": state.phase,
            "winner": state.winner,
            "updated_at": utc_now(),
            "pretraining": {run_id: summary.to_dict() for run_id, summary in summaries.items()},
            "pretraining_aggregates": pretraining_aggregates,
            "sft": sft_summaries,
            "runs": {
                run_id: {
                    "kind": run.spec.kind,
                    "variant": run.spec.variant,
                    "seed": run.spec.seed,
                    "role": run.spec.role,
                    "status": run.status,
                    "decision": run.decision,
                    "wandb_url": run.wandb_url,
                    "checkpoint_disposition": run.checkpoint_disposition,
                }
                for run_id, run in state.runs.items()
            },
        }
        payload["wandb_publish_errors"] = publish_summaries_to_wandb(
            spec,
            state,
            summaries,
            sft_summaries,
        )
        atomic_write_json(study_dir / "summary.json", payload)
        return payload


def publish_summaries_to_wandb(
    spec: StudySpec,
    state: StudyState,
    pretraining: Mapping[str, ConvergenceSummary],
    sft: Mapping[str, Any],
) -> list[str]:
    if os.environ.get("WANDB_MODE") in ("offline", "disabled") or not spec.wandb_entity:
        return []
    import wandb

    errors: list[str] = []
    api = None
    for run_id, run in state.runs.items():
        if run.wandb_run_id is None:
            continue
        summary_fields: dict[str, Any] = {
            "provenance/local_weight_disposition": run.checkpoint_disposition,
        }
        if run_id in pretraining:
            convergence = pretraining[run_id]
            summary_fields.update(
                {
                    "probe/peak_validation_accuracy": convergence.peak_accuracy,
                    "probe/final_validation_accuracy": convergence.final_accuracy,
                    "convergence/step_to_90": convergence.step_to_90,
                    "convergence/step_to_95": convergence.step_to_95,
                    "convergence/active_seconds_to_90": convergence.active_seconds_to_90,
                    "convergence/active_seconds_to_95": convergence.active_seconds_to_95,
                    "convergence/step_auc": convergence.step_auc,
                    "convergence/active_time_auc": convergence.active_time_auc,
                }
            )
        if run_id in sft.get("runs", {}):
            sft_summary = sft["runs"][run_id]
            summary_fields.update(
                {
                    "sft/peak_validation_accuracy": sft_summary["peak_accuracy"],
                    "sft/final_validation_accuracy": sft_summary["final_accuracy"],
                    "sft/test_accuracy": sft_summary["test_accuracy"],
                    "convergence/step_to_90": sft_summary["step_to_90"],
                    "convergence/step_to_95": sft_summary["step_to_95"],
                    "convergence/active_seconds_to_90": sft_summary["active_seconds_to_90"],
                    "convergence/active_seconds_to_95": sft_summary["active_seconds_to_95"],
                    "convergence/step_auc": sft_summary["step_auc"],
                    "convergence/active_time_auc": sft_summary["active_time_auc"],
                }
            )
        try:
            if api is None:
                api = wandb.Api()
            remote_run = api.run(f"{spec.wandb_entity}/{spec.wandb_project}/{run.wandb_run_id}")
            remote_run.summary.update(summary_fields)
            remote_run.update()
        except Exception as error:
            errors.append(f"{run_id}: {type(error).__name__}: {error}")
    return errors


def append_research_log(spec: StudySpec, summary: Mapping[str, Any], repo_root: Path) -> bool:
    log_path = repo_root / "research" / "LOG.md"
    marker = f"<!-- study:{spec.id}:phase:{summary['phase']} -->"
    existing = log_path.read_text() if log_path.is_file() else "# JEPA Research Log\n"
    if marker in existing:
        return False
    winner = summary.get("winner") or "none"
    variant_lines = [
        f"- `{variant.id}` mechanism: {variant.mechanism or 'not recorded'}; "
        f"changes: {', '.join(variant.changes) or 'not recorded'}"
        for variant in (spec.baseline, *spec.variants)
    ]
    run_lines = [
        f"- `{run_id}`: status={value['status']}, decision={value['decision']}, "
        f"W&B={value['wandb_url'] or 'unavailable'}"
        for run_id, value in summary["runs"].items()
    ]
    conclusion_by_phase = {
        "complete": f"{winner} completed confirmation and downstream evaluation.",
        "evaluation": f"{winner} passed confirmation; downstream evaluation is still running.",
        "not-confirmed": f"{winner} did not meet the three-seed confirmation rule.",
        "no-promotion": "No seed-0 candidate met a promotion threshold.",
    }
    conclusion = conclusion_by_phase.get(str(summary["phase"]), f"Study stopped in phase {summary['phase']}.")
    entry = (
        f"\n{marker}\n"
        f"## {spec.id}\n\n"
        f"- Question: {spec.question}\n"
        f"- Hypothesis: {spec.hypothesis}\n"
        "- Mechanisms and exact changes:\n" + "\n".join(variant_lines) + "\n"
        f"- Phase: {summary['phase']}\n"
        f"- Winner: {winner}\n"
        f"- Conclusion: {conclusion}\n"
        "- Follow-up: record interpretation and the next falsifiable hypothesis after metric review.\n"
        "- Checkpoint disposition: see each run below; deleted weights are not recoverable.\n\n"
        + "\n".join(run_lines)
        + "\n"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(entry)
    return True


def apply_rejected_retention(spec: StudySpec, repo_root: Path, *, study_close: bool = False) -> tuple[str, ...]:
    study_dir = study_directory(spec, repo_root)
    deleted: list[str] = []
    with StateStore(study_dir) as store:
        state = store.load()
        for run_id, run in state.runs.items():
            if run.decision == "rejected" or (study_close and run.decision == "retryable"):
                deleted.extend(
                    str(path) for path in cleanup_run_weights(state, run_id, study_dir, study_close=study_close)
                )
        store.save(state)
    return tuple(deleted)
