from __future__ import annotations

import hashlib
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import (
    ConvergenceSummary,
    MetricPoint,
    confirmation_decision,
    promotion_decision,
    rank_promoted_candidates,
    summarize_convergence,
)
from .models import WANDB_LOCAL_MODES, RunSpec, RunState, StudySpec, StudyState
from .runtime import (
    StateStore,
    append_locked_text,
    atomic_write_json,
    cleanup_run_weights,
    reconcile_state,
    study_directory,
    utc_now,
)


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_metric_points_file(
    metrics_path: Path, accuracy_key: str = "probe/validation_accuracy"
) -> tuple[MetricPoint, ...]:
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


def load_metric_points(run_dir: Path, accuracy_key: str = "probe/validation_accuracy") -> tuple[MetricPoint, ...]:
    return load_metric_points_file(run_dir / "metrics.jsonl", accuracy_key)


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
    repo_root: Path | None = None,
) -> dict[str, ConvergenceSummary]:
    points_by_run = _completed_pretrain_points(state)
    baseline_id = f"pretrain-{spec.baseline.id}-seed0"
    if spec.baseline_reference is not None:
        metrics_path = spec.baseline_reference.metrics
        if not metrics_path.is_absolute():
            metrics_path = (repo_root or Path.cwd()) / metrics_path
        reference_points = load_metric_points_file(metrics_path)
        if reference_points:
            points_by_run[baseline_id] = reference_points
    run_directories = {run_id: Path(run.run_dir) for run_id, run in state.runs.items() if run.run_dir is not None}
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
            cls_path_latency_median_ms=(
                _last_metric(run_directories[run_id], "diagnostics/cls_path_latency_median_ms")
                if run_id in run_directories
                else None
            ),
            cls_path_latency_p90_ms=(
                _last_metric(run_directories[run_id], "diagnostics/cls_path_latency_p90_ms")
                if run_id in run_directories
                else None
            ),
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
                "validation_peak_std": _sample_std(peak_values),
                "test_mean": statistics.mean(present_test_values),
                "test_std": _sample_std(present_test_values),
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
        final_active_times = [summary.active_seconds_at_step_horizon for summary in variant_summaries]
        cls_latencies = [summary.cls_path_latency_median_ms for summary in variant_summaries]
        present_cls_latencies = [value for value in cls_latencies if value is not None]
        present_times = [value for value in times if value is not None]
        all_reached_95 = len(present_times) == len(times)
        aggregates[variant] = {
            "peak_accuracy_mean": statistics.mean(peaks),
            "peak_accuracy_std": _sample_std(peaks),
            "active_time_auc_mean": statistics.mean(aucs),
            "active_time_auc_std": _sample_std(aucs),
            "active_seconds_to_95_mean": (statistics.mean(present_times) if all_reached_95 else None),
            "active_seconds_to_95_std": (_sample_std(present_times) if all_reached_95 else None),
            "censored_95_count": sum(value is None for value in times),
            "active_seconds_at_step_horizon_mean": statistics.mean(final_active_times),
            "active_seconds_at_step_horizon_std": _sample_std(final_active_times),
            "cls_path_latency_median_ms_mean": (
                statistics.mean(present_cls_latencies) if len(present_cls_latencies) == len(cls_latencies) else None
            ),
            "cls_path_latency_median_ms_std": (
                _sample_std(present_cls_latencies) if len(present_cls_latencies) == len(cls_latencies) else None
            ),
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
    for variant, role in ((spec.baseline.id, "baseline"), (winner, "winner")):
        finetune_config = spec.finetune_config_for(variant)
        if finetune_config is None:
            continue
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
                    config=finetune_config,
                    seed=seed,
                    role=role,
                    source_checkpoint=source_checkpoint,
                    shots_per_class=shots,
                    subset_seed=seed,
                    evaluate_test=True,
                )
                state.runs.setdefault(run_spec.id, RunState(run_spec))


def _passes_screening_control_gate(
    candidate_variant: str,
    candidate_summary: ConvergenceSummary,
    spec: StudySpec,
    summaries: Mapping[str, ConvergenceSummary],
) -> bool:
    control_variant = spec.promotion.screening_control_variant
    required_gain = spec.promotion.screening_control_accuracy_gain
    if control_variant is None or required_gain is None:
        return True
    if candidate_variant == control_variant:
        return False
    control_summary = summaries.get(f"pretrain-{control_variant}-seed0")
    if control_summary is None:
        return False
    return candidate_summary.peak_accuracy >= control_summary.peak_accuracy + required_gain


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
            control_gate_passes = _passes_screening_control_gate(
                run.spec.variant,
                candidate_summary,
                spec,
                summaries,
            )
            promoted = decision.promoted and control_gate_passes
            run.decision = "promoted" if promoted else "rejected"
            if promoted:
                qualifying.append((run.spec.variant, candidate_summary))
        if qualifying:
            winner = rank_promoted_candidates(qualifying)[0][0]
            state.winner = winner
            if spec.baseline_reference is not None:
                state.phase = "reference-promotion"
            else:
                _add_replication_runs(state, spec, winner)
                state.phase = "confirmation"
        elif state.phase == "screening":
            if spec.baseline_reference is not None:
                state.phase = "no-promotion"
            else:
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
        if seed_zero_decision.criterion is None or not _passes_screening_control_gate(
            state.winner,
            candidate_summaries[0],
            spec,
            summaries,
        ):
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
        summaries = calculate_study_summaries(state, spec, repo_root)
        sft_summaries = calculate_sft_summaries(state, spec)
        pretraining_aggregates = calculate_pretraining_aggregates(state, spec, summaries)
        advance_study(state, spec, summaries)
        store.save(state)
        payload = {
            "study_id": spec.id,
            "phase": state.phase,
            "winner": state.winner,
            "baseline_reference": (spec.baseline_reference.to_dict() if spec.baseline_reference is not None else None),
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
                    "attempt": run.attempt,
                    "started_at": run.started_at,
                    "finished_at": run.finished_at,
                    "terminal_event_id": run.terminal_event_id,
                    "run_dir": run.run_dir,
                    "error": run.error,
                    "wandb_url": run.wandb_url,
                    "checkpoint_disposition": run.checkpoint_disposition,
                }
                for run_id, run in state.runs.items()
            },
        }
        payload["detail_location"] = {
            "local_summary": str(study_dir / "summary.json"),
            "local_metrics": str(study_dir / "runs"),
            "external_tracker": False,
        }
        tracker_decision = spec.wandb_operation_decision("summary", os.environ.get("WANDB_MODE", "online"))
        payload["external_tracker"] = {
            "provider": "wandb" if spec.wandb_entity else None,
            "entity": spec.wandb_entity,
            "project": spec.wandb_project,
            "configured_authorization": spec.wandb_authorized,
            **tracker_decision.to_dict(),
        }
        payload["detail_location"]["external_tracker"] = tracker_decision.authorized
        payload["markdown_summary"] = markdown_summary(payload)
        atomic_write_json(study_dir / "summary.json", payload)
        payload["wandb_publish_errors"] = publish_summaries_to_wandb(
            spec,
            state,
            summaries,
            sft_summaries,
        )
        atomic_write_json(study_dir / "summary.json", payload)
    return payload


def markdown_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact, copyable terminal summary for research records."""
    lines = [
        f"Study: `{summary['study_id']}`",
        f"Phase: `{summary['phase']}`; winner: `{summary.get('winner') or 'none'}`",
        f"Detail: local artifacts at `{summary['detail_location']['local_summary']}`",
    ]
    for run_id, result in summary.get("pretraining", {}).items():
        lines.append(
            f"- `{run_id}`: peak={result.get('peak_accuracy')}, final={result.get('final_accuracy')}, "
            f"step_to_95={result.get('step_to_95')}, active_seconds_to_95={result.get('active_seconds_to_95')}, "
            f"active_seconds_at_step_horizon={result.get('active_seconds_at_step_horizon')}, "
            f"cls_path_latency_median_ms={result.get('cls_path_latency_median_ms')}"
        )
    return "\n".join(lines)


def publish_summaries_to_wandb(
    spec: StudySpec,
    state: StudyState,
    pretraining: Mapping[str, ConvergenceSummary],
    sft: Mapping[str, Any],
) -> list[str]:
    wandb_mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    decision = spec.wandb_operation_decision("summary", wandb_mode)
    if wandb_mode in WANDB_LOCAL_MODES or not spec.wandb_entity:
        return []
    if not spec.wandb_authorized:
        return ["W&B publication refused: study does not record explicit authorization"]
    if not spec.wandb_manifests_explicit:
        return ["W&B publication refused: emitted-data manifest is not explicit"]
    if decision.missing_data_classes:
        return [
            "W&B publication refused: approval is missing for emitted data classes: "
            f"{', '.join(decision.missing_data_classes)}"
        ]
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
                    "convergence/active_seconds_at_step_horizon": convergence.active_seconds_at_step_horizon,
                    "diagnostics/cls_path_latency_median_ms": convergence.cls_path_latency_median_ms,
                    "diagnostics/cls_path_latency_p90_ms": convergence.cls_path_latency_p90_ms,
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
    terminal_attempts = sorted(
        (run_id, int(value.get("attempt", 1)))
        for run_id, value in summary.get("runs", {}).items()
        if value.get("status") in ("completed", "failed", "timed_out")
    )
    operation_id = hashlib.sha256(
        json.dumps(
            {"study_id": spec.id, "phase": summary["phase"], "run_attempts": terminal_attempts},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:32]
    marker = f"<!-- study:{spec.id}:phase:{summary['phase']} -->"
    winner = summary.get("winner") or "none"
    variant_lines = []
    for variant in (spec.baseline, *spec.variants):
        mechanism = (variant.mechanism or "not recorded").rstrip(".")
        changes = ("; ".join(variant.changes) or "not recorded").rstrip(".")
        variant_lines.append(f"  - `{variant.id}`: Mechanism: {mechanism}. Changes: {changes}.")
    state = StateStore(study_directory(spec, repo_root)).load()
    provenance_lines: list[str] = []
    run_lines: list[str] = []
    pretraining = summary.get("pretraining", {})
    sft_runs = summary.get("sft", {}).get("runs", {})
    for run_id, value in summary["runs"].items():
        run = state.runs.get(run_id)
        if run is not None and run.run_dir is not None:
            provenance_path = Path(run.run_dir) / "provenance.json"
            if provenance_path.is_file():
                provenance = json.loads(provenance_path.read_text())
                repositories = []
                for repository_name in ("parent", "mjepa", "vit"):
                    repository = provenance.get(repository_name, {})
                    sha = repository.get("sha", "unknown")
                    branch = repository.get("branch", "unknown")
                    repositories.append(f"{repository_name}=`{sha}` (`{branch}`)")
                provenance_lines.append(f"  - `{run_id}`: " + ", ".join(repositories))

        metrics = pretraining.get(run_id) or sft_runs.get(run_id) or {}
        metric_parts: list[str] = []
        for key, digits in (
            ("peak_accuracy", 6),
            ("final_accuracy", 6),
            ("step_to_90", 0),
            ("step_to_95", 0),
            ("active_seconds_to_90", 3),
            ("active_seconds_to_95", 3),
            ("step_auc", 6),
            ("active_time_auc", 6),
            ("active_seconds_at_step_horizon", 3),
            ("cls_path_latency_median_ms", 6),
            ("cls_path_latency_p90_ms", 6),
            ("test_accuracy", 6),
        ):
            if key not in metrics:
                continue
            metric_value = metrics[key]
            rendered = "censored" if metric_value is None else f"{metric_value:.{digits}f}"
            metric_parts.append(f"{key}={rendered}")
        wandb_url = value.get("wandb_url")
        if wandb_url:
            wandb_reference = f"[run]({wandb_url})"
        elif run is not None and run.wandb_run_id:
            wandb_reference = f"offline/unlinked (`{run.wandb_run_id}`)"
        else:
            wandb_reference = "unavailable"
        run_lines.append(
            f"- `{run_id}`: attempt={value.get('attempt', 1)}; status={value['status']}; decision={value['decision']}; "
            f"started={value.get('started_at') or 'unknown'}; finished={value.get('finished_at') or 'unknown'}; "
            f"terminal_event={value.get('terminal_event_id') or 'unknown'}; "
            f"artifacts=`{value.get('run_dir') or 'unavailable'}`; "
            f"W&B={wandb_reference}; checkpoint={value['checkpoint_disposition']}; "
            f"metrics={', '.join(metric_parts) or 'unavailable'}; error={value.get('error') or 'none'}"
        )
    phase = str(summary["phase"])
    conclusion_by_phase = {
        "screening": "Seed-0 screening is still running.",
        "exploration": (
            "No initial seed-0 candidate met a promotion threshold; bounded seed-0 exploration is still running."
        ),
        "confirmation": f"{winner} passed seed-0 promotion; paired three-seed confirmation is still running.",
        "reference-promotion": (
            f"{winner} met a promotion threshold against the fixed seed-0 baseline reference; "
            "paired AdamW confirmation was outside this Muon-only sweep."
        ),
        "complete": f"{winner} completed confirmation and downstream evaluation.",
        "evaluation": f"{winner} passed confirmation; downstream evaluation is still running.",
        "not-confirmed": f"{winner} did not meet the three-seed confirmation rule.",
        "no-promotion": (
            "No seed-0 candidate met a promotion threshold."
            if spec.variants
            else "The baseline smoke run completed; no candidates were configured for promotion."
        ),
    }
    conclusion = conclusion_by_phase.get(phase, f"Study stopped in phase {phase}.")
    follow_up_by_phase = {
        "screening": "complete the preregistered seed-0 screening trials.",
        "exploration": "run the preregistered exploration trials.",
        "confirmation": "complete the paired baseline and winner replications.",
        "evaluation": "complete the paired supervised evaluations.",
    }
    follow_up = follow_up_by_phase.get(phase, "record interpretation and the next falsifiable hypothesis.")
    approved_data_classes = ", ".join(spec.wandb_approved_data_classes) or "none"
    local_summary = summary.get("detail_location", {}).get("local_summary", "unavailable")
    external_detail = summary.get("detail_location", {}).get("external_tracker", False)
    entry = (
        f"\n{marker}\n"
        f"## {spec.id}\n\n"
        f"- Question: {spec.question}\n"
        f"- Hypothesis: {spec.hypothesis}\n"
        "- Mechanisms and exact changes:\n" + "\n".join(variant_lines) + "\n"
        "- Launch code provenance:\n" + ("\n".join(provenance_lines) if provenance_lines else "  - unavailable") + "\n"
        f"- Phase: {summary['phase']}\n"
        f"- Winner: {winner}\n"
        f"- External tracker: provider={'W&B' if spec.wandb_entity else 'none'}; "
        f"account={spec.wandb_entity or 'none'}; project={spec.wandb_project if spec.wandb_entity else 'none'}; "
        f"authorized={spec.wandb_authorized}; approved_data_classes={approved_data_classes}\n"
        f"- Detail location: local summary and raw metrics under `{local_summary}`; "
        f"external_detail={external_detail}\n"
        f"- Conclusion: {conclusion}\n"
        f"- Follow-up: {follow_up}\n"
        "- Checkpoint disposition: see each run below; deleted weights are not recoverable.\n\n"
        + "\n".join(run_lines)
        + "\n"
    )
    return append_locked_text(log_path, entry, operation_id, initial_text="# Research Log\n")


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
