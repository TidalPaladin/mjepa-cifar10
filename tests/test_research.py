import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjepa_cifar10.research.cli import _load_spec, wandb_preflight_errors
from mjepa_cifar10.research.cli import main as research_main
from mjepa_cifar10.research.codex_notifications import (
    MANAGED_ROOT_MARKER_FILENAME,
    SweepResult,
    initialize_notification_root,
    queue_notification_from_terminal,
    write_notification_event,
)
from mjepa_cifar10.research.inventory import index_local_runs, index_wandb_runs, inventory_counts, open_inventory
from mjepa_cifar10.research.metrics import (
    ConvergenceSummary,
    MetricPoint,
    accuracy_auc,
    confirmation_decision,
    derive_convergence_targets,
    promotion_decision,
    summarize_convergence,
)
from mjepa_cifar10.research.models import (
    DEFAULT_MAX_PRETRAIN_TRIALS,
    WANDB_OPERATION_EMITTED_DATA_CLASSES,
    BaselineReference,
    PromotionRules,
    ResourceLimits,
    RunSpec,
    RunState,
    StudySpec,
    StudyState,
    VariantSpec,
)
from mjepa_cifar10.research.provenance import GitProvenance, assert_launch_provenance, collect_provenance
from mjepa_cifar10.research.runtime import (
    GIB,
    GPULock,
    StateStore,
    _pending_runs_in_study_order,
    append_locked_text,
    available_physical_gpus,
    build_managed_worker_environment,
    build_worker_environment,
    cleanup_run_weights,
    configure_lifecycle_environment,
    estimate_checkpoint_size,
    launch_available_runs,
    persist_terminal_and_queue_notification,
    prepare_retryable_runs,
    process_exit_error,
    reconcile_state,
    required_free_bytes,
    run_command_with_timeout,
    schedule_due_monitor_checks,
    schedule_monitor_check,
    validate_managed_paths,
)
from mjepa_cifar10.research.summary import (
    advance_study,
    append_research_log,
    calculate_study_summaries,
    publish_summaries_to_wandb,
)
from mjepa_cifar10.research.wake_context import WAKE_CONTEXT_FILENAME, WakeContext


def make_summary(
    *,
    peak: float,
    time_to_95: float | None,
    time_auc: float,
    active_seconds_at_horizon: float = 500.0,
    cls_latency_ms: float | None = None,
) -> ConvergenceSummary:
    return ConvergenceSummary(
        peak_accuracy=peak,
        final_accuracy=peak - 0.01,
        step_to_90=10,
        step_to_95=20 if time_to_95 is not None else None,
        active_seconds_to_90=50.0,
        active_seconds_to_95=time_to_95,
        step_auc=time_auc,
        active_time_auc=time_auc,
        step_horizon=100,
        active_time_horizon=500.0,
        active_seconds_at_step_horizon=active_seconds_at_horizon,
        cls_path_latency_median_ms=cls_latency_ms,
    )


def make_spec(tmp_path: Path, variants: int = 3) -> StudySpec:
    config = tmp_path / "config.yaml"
    config.write_text("trainer: test\n")
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return StudySpec(
        id="test-study",
        question="Does it help?",
        hypothesis="The change converges faster.",
        baseline=VariantSpec("baseline", config, "reference"),
        variants=tuple(VariantSpec(f"variant-{index}", config, "candidate") for index in range(variants)),
        data=data,
        log_root=tmp_path / "logs",
    )


def make_baseline_reference(tmp_path: Path) -> BaselineReference:
    metrics = tmp_path / "baseline-metrics.jsonl"
    metrics.write_text(
        "\n".join(
            (
                '{"_step": 10, "convergence/active_seconds": 50.0, "probe/validation_accuracy": 0.7}',
                '{"_step": 20, "convergence/active_seconds": 100.0, "probe/validation_accuracy": 0.8}',
            )
        )
        + "\n"
    )
    return BaselineReference(
        study_id="prior-study",
        run_id="pretrain-baseline-seed0",
        metrics=metrics,
        metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),
    )


def test_spec_loading_skips_dataset_validation_for_recovery_commands(mocker, tmp_path: Path) -> None:
    spec = replace(make_spec(tmp_path), data=Path("${CIFAR10_DATA}"))
    study_path = tmp_path / "study.yaml"
    study_path.touch()
    mocker.patch.object(StudySpec, "from_path", return_value=spec)
    args = argparse.Namespace(repo_root=tmp_path, study=study_path)

    loaded, repo_root = _load_spec(args, require_data=False)

    assert loaded is spec
    assert repo_root == tmp_path


def test_spec_loading_names_unset_dataset_variable_when_data_is_required(mocker, tmp_path: Path) -> None:
    spec = replace(make_spec(tmp_path), data=Path("${CIFAR10_DATA}"))
    study_path = tmp_path / "study.yaml"
    study_path.touch()
    mocker.patch.object(StudySpec, "from_path", return_value=spec)
    args = argparse.Namespace(repo_root=tmp_path, study=study_path)

    with pytest.raises(EnvironmentError, match=r"unset environment variable.*CIFAR10_DATA"):
        _load_spec(args, require_data=True)


def test_status_and_read_only_monitor_do_not_require_dataset(mocker, tmp_path: Path, capsys) -> None:
    spec = replace(make_spec(tmp_path), data=Path("${CIFAR10_DATA}"))
    study_path = tmp_path / "study.yaml"
    study_path.touch()
    mocker.patch.object(StudySpec, "from_path", return_value=spec)
    with StateStore(spec.log_root / spec.id) as store:
        store.load_or_create(spec, study_path)

    assert research_main(["status", str(study_path), "--repo-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["study_id"] == spec.id
    assert (
        research_main(
            [
                "monitor",
                str(study_path),
                "--repo-root",
                str(tmp_path),
                "--no-launch",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["study_id"] == spec.id


def test_convergence_targets_are_fixed_from_baseline_peak() -> None:
    assert derive_convergence_targets(0.8) == pytest.approx((0.72, 0.76))


def test_convergence_summary_reports_censoring_and_common_horizon_auc() -> None:
    points = (
        MetricPoint(step=10, active_seconds=100, accuracy=0.40),
        MetricPoint(step=20, active_seconds=200, accuracy=0.60),
        MetricPoint(step=30, active_seconds=300, accuracy=0.70),
    )

    summary = summarize_convergence(points, baseline_peak_accuracy=0.8, step_horizon=25, active_time_horizon=250)

    assert summary.step_to_90 is None
    assert summary.step_to_95 is None
    assert summary.step_auc == pytest.approx(accuracy_auc(points, "step", 25))
    assert summary.active_time_auc == pytest.approx(accuracy_auc(points, "active_seconds", 250))
    assert summary.active_seconds_at_step_horizon == pytest.approx(250.0)


@pytest.mark.parametrize(
    ("candidate", "criterion"),
    (
        (make_summary(peak=0.81, time_to_95=100, time_auc=0.50), "accuracy"),
        (make_summary(peak=0.795, time_to_95=84, time_auc=0.50), "time_to_95"),
        (make_summary(peak=0.795, time_to_95=100, time_auc=0.56), "time_auc"),
    ),
)
def test_promotion_rules_accept_each_documented_path(candidate: ConvergenceSummary, criterion: str) -> None:
    baseline = make_summary(peak=0.80, time_to_95=100, time_auc=0.50)

    decision = promotion_decision(baseline, candidate)

    assert decision.promoted
    assert decision.criterion == criterion


def test_opt_in_cost_promotion_requires_end_to_end_and_isolated_latency_gains() -> None:
    rules = PromotionRules(cost_gain=0.05)
    baseline = make_summary(
        peak=0.80,
        time_to_95=100,
        time_auc=0.50,
        active_seconds_at_horizon=500,
        cls_latency_ms=4.0,
    )
    candidate = make_summary(
        peak=0.795,
        time_to_95=100,
        time_auc=0.50,
        active_seconds_at_horizon=470,
        cls_latency_ms=2.0,
    )

    decision = promotion_decision(baseline, candidate, rules)

    assert decision.promoted
    assert decision.criterion == "cost"


@pytest.mark.parametrize(
    ("active_seconds_at_horizon", "cls_latency_ms"),
    ((480.0, 2.0), (470.0, 4.0), (470.0, None)),
)
def test_cost_promotion_rejects_incomplete_cost_gate(
    active_seconds_at_horizon: float,
    cls_latency_ms: float | None,
) -> None:
    rules = PromotionRules(cost_gain=0.05)
    baseline = make_summary(
        peak=0.80,
        time_to_95=100,
        time_auc=0.50,
        active_seconds_at_horizon=500,
        cls_latency_ms=4.0,
    )
    candidate = make_summary(
        peak=0.795,
        time_to_95=100,
        time_auc=0.50,
        active_seconds_at_horizon=active_seconds_at_horizon,
        cls_latency_ms=cls_latency_ms,
    )

    assert promotion_decision(baseline, candidate, rules).criterion is None


def test_confirmation_requires_mean_threshold_and_two_paired_improvements() -> None:
    baseline = [make_summary(peak=0.80, time_to_95=100, time_auc=0.5) for _ in range(3)]
    candidate = [
        make_summary(peak=0.82, time_to_95=100, time_auc=0.5),
        make_summary(peak=0.82, time_to_95=100, time_auc=0.5),
        make_summary(peak=0.80, time_to_95=100, time_auc=0.5),
    ]

    decision = confirmation_decision(baseline, candidate, "accuracy")

    assert decision.confirmed
    assert decision.paired_improvements == 2
    assert decision.mean_paired_difference == pytest.approx(0.0133333333)


def test_cost_confirmation_requires_two_joint_paired_improvements() -> None:
    rules = PromotionRules(cost_gain=0.05)
    baseline = [
        make_summary(
            peak=0.80,
            time_to_95=100,
            time_auc=0.5,
            active_seconds_at_horizon=500,
            cls_latency_ms=4.0,
        )
        for _ in range(3)
    ]
    candidate = [
        make_summary(
            peak=0.80,
            time_to_95=100,
            time_auc=0.5,
            active_seconds_at_horizon=active_seconds,
            cls_latency_ms=latency,
        )
        for active_seconds, latency in ((460, 2.0), (470, 3.0), (480, 4.1))
    ]

    decision = confirmation_decision(baseline, candidate, "cost", rules)

    assert decision.confirmed
    assert decision.paired_improvements == 2


def test_variant_specific_finetune_config_overrides_global_fallback(tmp_path: Path) -> None:
    pretrain_config = tmp_path / "pretrain.yaml"
    global_finetune = tmp_path / "global-finetune.yaml"
    single_cls_finetune = tmp_path / "single-cls-finetune.yaml"
    for config in (pretrain_config, global_finetune, single_cls_finetune):
        config.write_text("trainer: test\n")
    baseline = VariantSpec("baseline", pretrain_config, "reference")
    winner = VariantSpec(
        "winner",
        pretrain_config,
        "candidate",
        finetune_config=single_cls_finetune,
    )
    spec = replace(
        make_spec(tmp_path, variants=0),
        baseline=baseline,
        variants=(winner,),
        evaluation=replace(make_spec(tmp_path).evaluation, finetune_config=global_finetune),
    )

    assert spec.finetune_config_for("baseline") == global_finetune
    assert spec.finetune_config_for("winner") == single_cls_finetune


def test_study_rejects_more_than_eight_pretraining_trials(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    invalid = StudySpec(
        **{
            **spec.__dict__,
            "resources": ResourceLimits(max_pretraining_trials=DEFAULT_MAX_PRETRAIN_TRIALS + 1),
        }
    )

    with pytest.raises(ValueError, match="cannot exceed 8"):
        invalid.validate(require_files=False)


def test_screening_promotion_adds_only_four_replication_trials(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    now = "2026-01-01T00:00:00+00:00"
    state = StudyState(
        study_id=spec.id,
        spec_path="study.yaml",
        created_at=now,
        updated_at=now,
        runs={run.id: RunState(run, status="completed") for run in spec.initial_runs()},
    )
    baseline = make_summary(peak=0.80, time_to_95=100, time_auc=0.50)
    summaries = {f"pretrain-{spec.baseline.id}-seed0": baseline}
    for index in range(3):
        summaries[f"pretrain-variant-{index}-seed0"] = make_summary(
            peak=0.82 if index == 0 else 0.79,
            time_to_95=100,
            time_auc=0.50,
        )

    advance_study(state, spec, summaries)

    assert state.phase == "confirmation"
    assert state.winner == "variant-0"
    assert sum(run.spec.kind == "pretrain" for run in state.runs.values()) == 8


def test_screening_control_gate_requires_peak_gain_before_cost_promotion(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        promotion=PromotionRules(
            accuracy_gain=1.0,
            convergence_gain=0.99,
            auc_gain=1.0,
            maximum_accuracy_loss=0.04,
            cost_gain=0.05,
            screening_control_variant="variant-0",
            screening_control_accuracy_gain=0.05,
        ),
    )
    now = "2026-01-01T00:00:00+00:00"
    state = StudyState(
        study_id=spec.id,
        spec_path="study.yaml",
        created_at=now,
        updated_at=now,
        runs={run.id: RunState(run, status="completed") for run in spec.initial_runs()},
    )
    summaries = {
        "pretrain-baseline-seed0": make_summary(
            peak=0.91,
            time_to_95=100,
            time_auc=0.50,
            active_seconds_at_horizon=500,
            cls_latency_ms=4.0,
        ),
        "pretrain-variant-0-seed0": make_summary(
            peak=0.83,
            time_to_95=None,
            time_auc=0.45,
            active_seconds_at_horizon=450,
            cls_latency_ms=2.0,
        ),
        "pretrain-variant-1-seed0": make_summary(
            peak=0.879,
            time_to_95=None,
            time_auc=0.46,
            active_seconds_at_horizon=450,
            cls_latency_ms=2.0,
        ),
        "pretrain-variant-2-seed0": make_summary(
            peak=0.885,
            time_to_95=None,
            time_auc=0.47,
            active_seconds_at_horizon=450,
            cls_latency_ms=2.0,
        ),
    }

    advance_study(state, spec, summaries)

    assert state.phase == "confirmation"
    assert state.winner == "variant-2"
    assert state.runs["pretrain-variant-0-seed0"].decision == "rejected"
    assert state.runs["pretrain-variant-1-seed0"].decision == "rejected"
    assert state.runs["pretrain-variant-2-seed0"].decision == "promoted"


@pytest.mark.parametrize(
    "rules",
    (
        PromotionRules(screening_control_variant="variant-0"),
        PromotionRules(screening_control_accuracy_gain=0.05),
        PromotionRules(screening_control_variant="missing", screening_control_accuracy_gain=0.05),
        PromotionRules(screening_control_variant="baseline", screening_control_accuracy_gain=0.05),
        PromotionRules(screening_control_variant="variant-0", screening_control_accuracy_gain=0.0),
    ),
)
def test_study_rejects_invalid_screening_control_gate(tmp_path: Path, rules: PromotionRules) -> None:
    spec = replace(make_spec(tmp_path), promotion=rules)

    with pytest.raises(ValueError, match="screening control"):
        spec.validate(require_files=False)


def test_reference_baseline_schedules_only_muon_candidates(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path, variants=5),
        baseline_reference=make_baseline_reference(tmp_path),
        resources=ResourceLimits(max_pretraining_trials=5),
    )

    initial_runs = spec.initial_runs()

    assert [run.variant for run in initial_runs] == [f"variant-{index}" for index in range(5)]
    assert all(run.role == "candidate" for run in initial_runs)


def test_reference_baseline_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    reference = make_baseline_reference(tmp_path)
    spec = replace(
        make_spec(tmp_path, variants=1),
        baseline_reference=reference,
        resources=ResourceLimits(max_pretraining_trials=1),
    )
    reference.metrics.write_text("changed\n")

    with pytest.raises(ValueError, match="baseline reference metrics hash mismatch"):
        spec.validate(relative_to=tmp_path)


def test_reference_baseline_participates_in_common_horizon_summary(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path, variants=1),
        baseline_reference=make_baseline_reference(tmp_path),
        resources=ResourceLimits(max_pretraining_trials=1),
    )
    candidate = spec.initial_runs()[0]
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            (
                '{"_step": 10, "convergence/active_seconds": 40.0, "probe/validation_accuracy": 0.75}',
                '{"_step": 20, "convergence/active_seconds": 90.0, "probe/validation_accuracy": 0.81}',
            )
        )
        + "\n"
    )
    now = "2026-01-01T00:00:00+00:00"
    state = StudyState(
        study_id=spec.id,
        spec_path="study.yaml",
        created_at=now,
        updated_at=now,
        runs={candidate.id: RunState(candidate, status="completed", run_dir=str(run_dir))},
    )

    summaries = calculate_study_summaries(state, spec, repo_root=tmp_path)

    assert set(summaries) == {"pretrain-baseline-seed0", candidate.id}
    assert summaries["pretrain-baseline-seed0"].peak_accuracy == pytest.approx(0.8)
    assert summaries[candidate.id].active_time_horizon == pytest.approx(90.0)


def test_reference_baseline_promotion_stops_without_adamw_replication(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path, variants=3),
        baseline_reference=make_baseline_reference(tmp_path),
        resources=ResourceLimits(max_pretraining_trials=3),
    )
    now = "2026-01-01T00:00:00+00:00"
    state = StudyState(
        study_id=spec.id,
        spec_path="study.yaml",
        created_at=now,
        updated_at=now,
        runs={run.id: RunState(run, status="completed") for run in spec.initial_runs()},
    )
    summaries = {"pretrain-baseline-seed0": make_summary(peak=0.8, time_to_95=100, time_auc=0.5)}
    summaries.update(
        {
            run.spec.id: make_summary(
                peak=0.82 if run.spec.variant == "variant-0" else 0.79,
                time_to_95=100,
                time_auc=0.5,
            )
            for run in state.runs.values()
        }
    )

    advance_study(state, spec, summaries)

    assert state.phase == "reference-promotion"
    assert state.winner == "variant-0"
    assert len(state.runs) == 3
    assert all(run.spec.variant != spec.baseline.id for run in state.runs.values())


def test_gpu_lock_excludes_duplicate_scheduler_assignment(tmp_path: Path) -> None:
    with GPULock(1, tmp_path):
        assert available_physical_gpus((1, 2), lock_root=tmp_path) == (2,)


def test_pending_runs_follow_study_order_after_sorted_state_round_trip(tmp_path: Path) -> None:
    base_spec = make_spec(tmp_path)
    config = base_spec.baseline.config
    spec = replace(
        base_spec,
        variants=(
            VariantSpec("muon-matched", config, "matched-rate Muon"),
            VariantSpec("muon-lr-half", config, "half-rate Muon"),
            VariantSpec("adamw-lr-half", config, "half-rate AdamW"),
        ),
    )
    expected_runs = spec.initial_runs()
    sorted_runs = sorted(expected_runs, key=lambda run: run.id)
    now = "2026-01-01T00:00:00+00:00"
    state = StudyState(
        study_id=spec.id,
        spec_path="study.yaml",
        created_at=now,
        updated_at=now,
        runs={run.id: RunState(run) for run in sorted_runs},
    )

    ordered = _pending_runs_in_study_order(state, spec)

    assert [run.spec.id for run in ordered] == [run.id for run in expected_runs]


def test_prepare_retryable_runs_reconciles_terminal_state_first(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, variants=0)
    run_spec = spec.initial_runs()[0]
    run_dir = tmp_path / "logs" / spec.id / "runs" / run_spec.id
    run_dir.mkdir(parents=True)
    (run_dir / "terminal.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "exit_code": -15,
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:01:00+00:00",
                "attempt": 1,
                "terminal_event_id": "terminal-event",
            }
        )
    )
    now = "2026-01-01T00:00:00+00:00"
    run = RunState(run_spec, status="launching", run_dir=str(run_dir), attempt=1)
    state = StudyState(
        study_id=spec.id,
        spec_path="study.yaml",
        created_at=now,
        updated_at=now,
        runs={run.spec.id: run},
    )

    retry_count = prepare_retryable_runs(state)

    assert retry_count == 1
    assert run.status == "pending"
    assert run.attempt == 2
    assert not (run_dir / "terminal.json").exists()


def test_managed_launch_persists_immutable_wake_context_before_spawn(
    mocker,
    tmp_path: Path,
) -> None:
    spec = make_spec(tmp_path, variants=0)
    study_path = tmp_path / "study.yaml"
    study_path.write_text("id: test-study\n")
    wake_context = WakeContext(
        thread_id="thread-a",
        permission_profile=":danger-full-access",
        approval_policy="never",
        captured_at=datetime(2026, 7, 23, tzinfo=UTC),
    )
    mocker.patch("mjepa_cifar10.research.runtime.assert_storage_available")
    mocker.patch(
        "mjepa_cifar10.research.runtime.available_physical_gpus",
        return_value=(1,),
    )

    def spawn(*_args, **_kwargs):
        context_path = tmp_path / "logs" / spec.id / "runs" / spec.initial_runs()[0].id / WAKE_CONTEXT_FILENAME
        assert context_path.is_file()
        return SimpleNamespace(pid=1234)

    mocker.patch("mjepa_cifar10.research.runtime.subprocess.Popen", side_effect=spawn)

    state = launch_available_runs(
        spec,
        study_path,
        tmp_path,
        development=True,
        lock_root=tmp_path / "locks",
        wake_context=wake_context,
    )

    launched = state.runs[spec.initial_runs()[0].id]
    assert launched.originating_thread_id == wake_context.thread_id


def test_timeout_terminates_process_group(mocker, tmp_path: Path) -> None:
    process = mocker.Mock(pid=1234)
    process.wait.side_effect = [subprocess.TimeoutExpired("train", 1), 0]
    process.poll.return_value = None
    mocker.patch("mjepa_cifar10.research.runtime.subprocess.Popen", return_value=process)
    kill_group = mocker.patch("mjepa_cifar10.research.runtime.os.killpg")
    environment = {"MJEPA_RESEARCH_REPO_ROOT": str(tmp_path)}

    with (tmp_path / "run.log").open("w") as log_file:
        exit_code, timed_out = run_command_with_timeout(
            ("train",),
            env=environment,
            timeout_seconds=1,
            log_file=log_file,
        )

    assert (exit_code, timed_out) == (124, True)
    kill_group.assert_called_once()


def test_process_exit_error_records_timeout_exit_code_and_signal() -> None:
    assert process_exit_error(0, timed_out=False) is None
    assert process_exit_error(124, timed_out=True) == "24-hour job timeout exceeded"
    assert process_exit_error(3, timed_out=False) == "training process exited with status 3"
    assert process_exit_error(-9, timed_out=False) == "training process terminated by SIGKILL"


def test_worker_environment_drops_inherited_wandb_service_socket(tmp_path: Path) -> None:
    environment = build_worker_environment(
        {"WANDB_SERVICE": "dead-socket-token", "WANDB_MODE": "offline"},
        physical_gpu=1,
        repo_root=tmp_path,
    )

    assert "WANDB_SERVICE" not in environment
    assert environment["WANDB_MODE"] == "offline"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_worker_environment_records_managed_lifecycle_identity(tmp_path: Path) -> None:
    environment = configure_lifecycle_environment(
        {},
        study_id="study-a",
        run_id="run-a",
        attempt=2,
        originating_thread_id="thread-a",
    )

    assert environment == {
        "MJEPA_RESEARCH_STUDY_ID": "study-a",
        "MJEPA_RESEARCH_RUN_ID": "run-a",
        "MJEPA_RESEARCH_ATTEMPT": "2",
        "MJEPA_RESEARCH_THREAD_ID": "thread-a",
    }


def test_worker_environment_forces_offline_without_complete_wandb_approval(tmp_path: Path) -> None:
    partial_approval = replace(
        make_spec(tmp_path),
        wandb_entity="entity",
        wandb_authorized=True,
        wandb_approved_data_classes=("metrics",),
    )
    full_approval = replace(
        partial_approval,
        wandb_approved_data_classes=("metrics", "configs", "provenance"),
    )
    implicit_manifest = replace(full_approval, wandb_manifests_explicit=False)

    forced_offline = build_managed_worker_environment({"WANDB_MODE": "online"}, 1, tmp_path, partial_approval)
    authorized_online = build_managed_worker_environment({"WANDB_MODE": "online"}, 1, tmp_path, full_approval)
    implicit_manifest_offline = build_managed_worker_environment(
        {"WANDB_MODE": "online"}, 1, tmp_path, implicit_manifest
    )

    assert forced_offline["WANDB_MODE"] == "offline"
    assert authorized_online["WANDB_MODE"] == "online"
    assert implicit_manifest_offline["WANDB_MODE"] == "offline"


def test_heartbeat_callback_is_called_while_worker_process_runs(mocker, tmp_path: Path) -> None:
    process = mocker.Mock(pid=1234)
    process.poll.side_effect = [None, 0]
    process.wait.return_value = None
    mocker.patch("mjepa_cifar10.research.runtime.subprocess.Popen", return_value=process)
    heartbeat = mocker.Mock()
    environment = {"MJEPA_RESEARCH_REPO_ROOT": str(tmp_path)}

    with (tmp_path / "run.log").open("w") as log_file:
        exit_code, timed_out = run_command_with_timeout(
            ("train",),
            env=environment,
            timeout_seconds=10,
            log_file=log_file,
            heartbeat_callback=heartbeat,
        )

    assert (exit_code, timed_out) == (0, False)
    heartbeat.assert_called_once()


def test_heartbeat_failure_terminates_child_before_propagating(mocker, tmp_path: Path) -> None:
    process = mocker.Mock(pid=1234)
    process.poll.return_value = None
    mocker.patch("mjepa_cifar10.research.runtime.subprocess.Popen", return_value=process)
    terminate = mocker.patch("mjepa_cifar10.research.runtime._terminate_process_group")
    heartbeat = mocker.Mock(side_effect=RuntimeError("heartbeat disk failure"))
    environment = {"MJEPA_RESEARCH_REPO_ROOT": str(tmp_path)}

    with (tmp_path / "run.log").open("w") as log_file:
        with pytest.raises(RuntimeError, match="heartbeat disk failure"):
            run_command_with_timeout(
                ("train",),
                env=environment,
                timeout_seconds=10,
                log_file=log_file,
                heartbeat_callback=heartbeat,
            )

    terminate.assert_called_once_with(process)


def test_research_log_append_is_operation_deduplicated(tmp_path: Path) -> None:
    log_path = tmp_path / "research" / "LOG.md"

    assert append_locked_text(log_path, "first\n", "operation-1")
    assert not append_locked_text(log_path, "first\n", "operation-1")

    content = log_path.read_text()
    assert '"operation_id":"operation-1"' in content
    assert content.endswith("first\n")


def test_research_log_rejects_operation_collisions_and_metadata_injection(tmp_path: Path) -> None:
    log_path = tmp_path / "research" / "LOG.md"
    assert append_locked_text(log_path, "first\n", "operation-1", initial_text="# Log\n")

    with pytest.raises(ValueError, match="different content"):
        append_locked_text(log_path, "changed\n", "operation-1", initial_text="# Log\n")
    with pytest.raises(ValueError, match="reserved metadata"):
        append_locked_text(log_path, "<!-- autoresearch-operation:forged -->\n", "operation-2")

    malformed_path = tmp_path / "research" / "MALFORMED.md"
    malformed_path.write_text(
        '<!-- autoresearch-operation:{"content_sha256":7,"operation_id":"forged"} -->\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid operation metadata"):
        append_locked_text(malformed_path, "entry\n", "operation-2")


def test_concurrent_first_research_log_writes_create_one_header(tmp_path: Path) -> None:
    log_path = tmp_path / "research" / "LOG.md"

    def append(index: int) -> bool:
        return append_locked_text(log_path, f"entry {index}\n", f"operation-{index}", initial_text="# Log\n")

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(append, range(8)))

    content = log_path.read_text(encoding="utf-8")
    assert content.count("# Log\n") == 1
    for index in range(8):
        assert content.count(f"entry {index}\n") == 1


def test_managed_path_validation_rejects_repository_root(tmp_path: Path) -> None:
    spec = replace(make_spec(tmp_path), log_root=tmp_path)

    with pytest.raises(ValueError, match="must not be the repository root"):
        validate_managed_paths(spec, tmp_path)


def test_monitor_schedule_uses_two_startup_checks_then_steady_state(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    run = RunState(spec.initial_runs()[0], status="running")
    check_time = datetime(2026, 1, 1, tzinfo=UTC)

    schedule_monitor_check(run, now=check_time)
    assert run.last_check_interval_seconds == 600
    schedule_monitor_check(run, now=check_time)
    assert run.last_check_interval_seconds == 1800
    schedule_monitor_check(run, now=check_time)
    assert run.last_check_interval_seconds == 1800


def test_monitor_reschedules_only_due_runs_and_clears_terminal_poll(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    check_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    due = RunState(
        spec.initial_runs()[0],
        status="running",
        next_check_at="2026-01-01T11:59:00+00:00",
    )
    future = RunState(
        replace(spec.initial_runs()[0], id="future"),
        status="running",
        next_check_at="2026-01-01T13:00:00+00:00",
    )
    terminal = RunState(
        replace(spec.initial_runs()[0], id="terminal"),
        status="completed",
        next_check_at="2026-01-01T12:30:00+00:00",
        next_check_reason="steady-state-check",
    )

    assert schedule_due_monitor_checks((due, future, terminal), now=check_time)

    assert due.routine_check_count == 1
    assert future.routine_check_count == 0
    assert future.next_check_at == "2026-01-01T13:00:00+00:00"
    assert terminal.next_check_at is None
    assert terminal.next_check_reason == "terminal"


def test_state_recovers_terminal_worker_file(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    study_dir = tmp_path / "study"
    with StateStore(study_dir) as store:
        state = store.load_or_create(spec, tmp_path / "study.yaml")
        run = next(iter(state.runs.values()))
        run_dir = study_dir / "runs" / run.spec.id
        run_dir.mkdir(parents=True)
        run.run_dir = str(run_dir)
        run.status = "running"
        store.save(state)
    (run_dir / "terminal.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "started_at": "start",
                "finished_at": "finish",
                "wandb_run_id": "abc123",
            }
        )
    )

    with StateStore(study_dir) as store:
        recovered = store.load()
        assert reconcile_state(recovered)
        store.save(recovered)

    assert recovered.runs[run.spec.id].status == "completed"
    assert recovered.runs[run.spec.id].wandb_run_id == "abc123"


def test_state_recovers_accepted_notification_metadata(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    study_dir = tmp_path / "logs" / spec.id
    run = RunState(spec.initial_runs()[0], status="running")
    run_dir = study_dir / "runs" / run.spec.id
    run_dir.mkdir(parents=True)
    run.run_dir = str(run_dir)
    terminal_path = run_dir / "terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "started_at": "2026-07-20T11:00:00+00:00",
                "finished_at": "2026-07-20T12:00:00+00:00",
                "wandb_run_id": "abc123",
                "attempt": 1,
                "terminal_event_id": "12345678-1234-5678-9234-567812345678",
                "originating_thread_id": "thread-1",
            }
        )
    )
    event = queue_notification_from_terminal(
        terminal_path,
        study_dir.parent,
        study_id=spec.id,
        run_id=run.spec.id,
    ).with_acceptance(
        accepted_at=datetime(2026, 7, 20, 12, 1, tzinfo=UTC),
        rpc_method="turn/start",
        turn_id="turn-1",
    )
    write_notification_event(event, study_dir.parent)
    state = StudyState(spec.id, "study.yaml", "now", "now", {run.spec.id: run})

    assert reconcile_state(state)

    assert run.status == "completed"
    assert run.notification_state == "accepted"
    assert run.notification_attempts == 1
    assert run.notification_accepted_rpc_method == "turn/start"
    assert run.notification_accepted_turn_id == "turn-1"


def test_terminal_result_survives_notification_queue_failure(mocker, tmp_path: Path) -> None:
    terminal_path = tmp_path / "logs" / "study" / "runs" / "run" / "terminal.json"
    terminal = {
        "status": "completed",
        "exit_code": 0,
        "started_at": "2026-07-20T11:00:00+00:00",
        "finished_at": "2026-07-20T12:00:00+00:00",
        "attempt": 1,
        "terminal_event_id": "12345678-1234-5678-9234-567812345678",
        "originating_thread_id": "thread-1",
    }
    mocker.patch(
        "mjepa_cifar10.research.codex_notifications.queue_notification_from_terminal",
        side_effect=RuntimeError("app-server queue unavailable"),
    )

    notification_error = persist_terminal_and_queue_notification(
        terminal_path,
        terminal,
        tmp_path / "logs",
        study_id="study",
        run_id="run",
    )

    assert json.loads(terminal_path.read_text())["status"] == "completed"
    assert "app-server queue unavailable" in (notification_error or "")
    assert not terminal_path.with_name("notification.json").exists()


def test_notify_worker_defaults_to_direct_daemon_socket(mocker, tmp_path: Path, capsys) -> None:
    initialize_notification_root(tmp_path)
    socket_path = tmp_path / "app-server.sock"
    resolve_socket = mocker.patch(
        "mjepa_cifar10.research.cli.resolve_event_controller_socket",
        return_value=socket_path,
    )

    exit_code = research_main(
        [
            "notify-worker",
            "--once",
            "--root",
            str(tmp_path),
            "--study-id",
            "study-a",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["discovered"] == 0
    resolve_socket.assert_called_once_with(None)


def test_event_controller_persists_startup_and_sweep_output(mocker, tmp_path: Path, capsys) -> None:
    initialize_notification_root(tmp_path)
    (tmp_path / "study-a").mkdir()
    socket_path = tmp_path / "app-server.sock"
    mocker.patch(
        "mjepa_cifar10.research.cli.sweep_notifications",
        new=mocker.AsyncMock(return_value=SweepResult(discovered=1, due=1, accepted=1)),
    )

    def serve_once(_root, **kwargs) -> None:
        kwargs["deliver"]()

    mocker.patch("mjepa_cifar10.research.cli.serve_controller", side_effect=serve_once)

    assert (
        research_main(
            [
                "event-controller",
                "--root",
                str(tmp_path),
                "--socket",
                str(socket_path),
                "--study-id",
                "study-a",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["accepted"] == 1
    log_paths = tuple((tmp_path / ".event-controller").glob("*.jsonl"))
    assert len(log_paths) == 1
    records = [json.loads(line) for line in log_paths[0].read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "controller_started",
        "notification_sweep",
        "controller_stopped",
    ]
    assert records[1]["accepted"] == 1


def test_state_marks_missing_supervisor_as_retryable(mocker, tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    study_dir = tmp_path / "study"
    with StateStore(study_dir) as store:
        state = store.load_or_create(spec, tmp_path / "study.yaml")
    run = next(iter(state.runs.values()))
    run.status = "running"
    run.pid = 4242
    run.run_dir = str(study_dir / "runs" / run.spec.id)
    mocker.patch("mjepa_cifar10.research.runtime._pid_is_alive", return_value=False)

    assert reconcile_state(state)

    assert run.status == "failed"
    assert run.decision == "retryable"
    assert "without writing terminal" in (run.error or "")


def test_retry_resets_only_retryable_terminal_runs(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    runs = {run.id: RunState(run) for run in spec.initial_runs()}
    retryable = next(iter(runs.values()))
    retryable.status = "failed"
    retryable.decision = "retryable"
    retryable.attempt = 2
    retryable.pid = 123
    retryable_run_dir = tmp_path / "runs" / retryable.spec.id
    retryable_run_dir.mkdir(parents=True)
    retryable.run_dir = str(retryable_run_dir)
    (retryable_run_dir / "terminal.json").write_text('{"status": "failed"}')
    for artifact_name in (
        "worker.json",
        "notification.json",
        "progress.json",
        "first-cycle.json",
        "first-cycle.notification.json",
        "supervisor-lost.json",
        "supervisor-lost.notification.json",
        "progress-stalled.json",
        "progress-stalled.notification.json",
    ):
        (retryable_run_dir / artifact_name).write_text("{}")
    retained = list(runs.values())[1]
    retained.status = "completed"
    retained.decision = "rejected"
    state = StudyState(spec.id, "study.yaml", "now", "now", runs)

    assert prepare_retryable_runs(state) == 1

    assert retryable.status == "pending"
    assert retryable.decision == "pending"
    assert retryable.pid is None
    assert retryable.attempt == 3
    assert not (retryable_run_dir / "terminal.json").exists()
    assert not any(
        (retryable_run_dir / artifact_name).exists()
        for artifact_name in (
            "worker.json",
            "notification.json",
            "progress.json",
            "first-cycle.json",
            "first-cycle.notification.json",
            "supervisor-lost.json",
            "supervisor-lost.notification.json",
            "progress-stalled.json",
            "progress-stalled.notification.json",
        )
    )
    assert len(tuple((retryable_run_dir / "attempts").glob("*.json"))) == 10
    assert retained.status == "completed"


def test_guarded_cleanup_only_deletes_terminal_rejected_managed_run(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    run_id = "pretrain-rejected-seed0"
    run_dir = study_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.pt"
    backbone = run_dir / "backbone.safetensors"
    checkpoint.write_bytes(b"full-checkpoint")
    backbone.write_bytes(b"backbone")
    run_spec = RunSpec(run_id, "pretrain", "rejected", Path("config.yaml"), 0, "candidate")
    run = RunState(run_spec, status="completed", decision="rejected", run_dir=str(run_dir))
    state = StudyState("study", "study.yaml", "now", "now", {run_id: run})

    deleted = cleanup_run_weights(state, run_id, study_dir)

    assert deleted == (checkpoint,)
    assert not checkpoint.exists()
    assert backbone.exists()
    assert run.checkpoint_disposition == "deleted-not-recoverable"
    retention_records = [json.loads(line) for line in (study_dir / "retention.jsonl").read_text().splitlines()]
    assert [record["phase"] for record in retention_records] == ["planned", "deleted"]
    assert retention_records[0]["bytes_planned"] == len(b"full-checkpoint")


def test_cleanup_rejects_path_outside_exact_managed_run_directory(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    outside = tmp_path / "legacy"
    outside.mkdir()
    run_spec = RunSpec("run", "pretrain", "variant", Path("config.yaml"), 0, "candidate")
    run = RunState(run_spec, status="completed", decision="rejected", run_dir=str(outside))
    state = StudyState("study", "study.yaml", "now", "now", {"run": run})

    with pytest.raises(ValueError, match="outside exact managed"):
        cleanup_run_weights(state, "run", study_dir)


def test_storage_requirement_includes_atomic_replacement_space() -> None:
    assert required_free_bytes(50, concurrent_jobs=2, estimated_checkpoint_size=3 * GIB) == 62 * GIB


def test_checkpoint_estimate_uses_only_matching_model_class(tmp_path: Path) -> None:
    matching = tmp_path / "matching"
    other = tmp_path / "other"
    matching.mkdir()
    other.mkdir()
    (matching / "metadata.json").write_text('{"model_class": "vit-small"}')
    (other / "metadata.json").write_text('{"model_class": "vit-tiny"}')
    (matching / "checkpoint.pt").write_bytes(b"small")
    (other / "checkpoint.pt").write_bytes(b"much-larger-checkpoint")

    assert estimate_checkpoint_size((tmp_path,), fallback_gib=3, model_class="vit-small") == len(b"small")


def test_provenance_errors_refuse_launch(mocker, tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    mocker.patch(
        "mjepa_cifar10.research.provenance.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    mocker.patch(
        "mjepa_cifar10.research.provenance.collect_provenance",
        return_value=SimpleNamespace(errors=("parent repository is dirty",)),
    )

    with pytest.raises(RuntimeError, match="parent repository is dirty"):
        assert_launch_provenance(spec, tmp_path)


def test_provenance_accepts_clean_unpushed_tandem_commits(mocker, tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        code_shas={
            "parent": "parent-sha",
            "mjepa": "mjepa-sha",
            "vit": "vit-sha",
        },
    )
    (tmp_path / "uv.lock").write_text("locked\n")
    mocker.patch(
        "mjepa_cifar10.research.provenance.git_provenance",
        side_effect=(
            GitProvenance(
                str(tmp_path),
                "parent-sha",
                "codex/research/test-study",
                False,
                "origin/codex/research/test-study",
                True,
            ),
            GitProvenance(
                str(tmp_path.parent / "mjepa"),
                "mjepa-sha",
                "codex/research/test-study",
                False,
                None,
                False,
            ),
            GitProvenance(
                str(tmp_path.parent / "vit"),
                "vit-sha",
                "codex/research/test-study",
                False,
                None,
                False,
            ),
        ),
    )
    mocker.patch(
        "mjepa_cifar10.research.provenance._installed_source",
        side_effect=lambda name: {"vcs_info": {"commit_id": f"{name}-sha"}},
    )

    report = collect_provenance(spec, tmp_path)

    assert report.errors == ()


def test_provenance_still_requires_pushed_parent_and_exact_tandem_head(mocker, tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        code_shas={
            "parent": "parent-sha",
            "mjepa": "mjepa-sha",
            "vit": "vit-sha",
        },
    )
    (tmp_path / "uv.lock").write_text("locked\n")
    mocker.patch(
        "mjepa_cifar10.research.provenance.git_provenance",
        side_effect=(
            GitProvenance(str(tmp_path), "parent-sha", "codex/research/test-study", False, None, False),
            GitProvenance(str(tmp_path.parent / "mjepa"), "mjepa-sha", "codex/research/test-study", False, None, False),
            GitProvenance(
                str(tmp_path.parent / "vit"),
                "other-vit-sha",
                "codex/research/test-study",
                False,
                None,
                False,
            ),
        ),
    )
    mocker.patch(
        "mjepa_cifar10.research.provenance._installed_source",
        side_effect=lambda name: {"vcs_info": {"commit_id": f"{name}-sha"}},
    )

    errors = collect_provenance(spec, tmp_path).errors

    assert "parent branch is not pushed at its current SHA" in errors
    assert "vit SHA mismatch: expected vit-sha, got other-vit-sha" in errors


def test_inventory_combines_config_metrics_packages_and_weight_availability(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "legacy-run"
    wandb_files = run_dir / "wandb" / "run-abc" / "files"
    wandb_files.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("trainer: legacy\n")
    (run_dir / "metrics.jsonl").write_text('{"_step": 1, "val/acc": 0.5}\n')
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    (wandb_files / "requirements.txt").write_text("torch==2.13.0\n")
    (wandb_files / "wandb-summary.json").write_text('{"val/acc": 0.5}')

    with closing(open_inventory(tmp_path / "inventory.sqlite3")) as connection:
        assert index_local_runs(tmp_path, connection) == 1
        assert inventory_counts(connection) == {"legacy": 1}
        row = connection.execute(
            "SELECT config_json, history_json, packages_json, checkpoint_available FROM runs"
        ).fetchone()

    assert row is not None
    assert "trainer: legacy" in row[0]
    assert "val/acc" in row[1]
    assert "torch==2.13.0" in row[2]
    assert row[3] == 1


def test_wandb_inventory_passes_history_keys_as_a_list(mocker, tmp_path: Path) -> None:
    run = SimpleNamespace(
        id="wandb-run",
        url="https://wandb.example/run",
        config={},
        summary={},
        metadata={},
        scan_history=mocker.Mock(return_value=[]),
    )
    api = mocker.Mock()
    api.runs.return_value = [run]
    mocker.patch("wandb.Api", return_value=api)

    with closing(open_inventory(tmp_path / "inventory.sqlite3")) as connection:
        assert index_wandb_runs("entity", "project", connection) == 1

    history_keys = run.scan_history.call_args.kwargs["keys"]
    assert isinstance(history_keys, list)
    assert history_keys[0] == "_step"


def test_summary_publishes_standardized_fields_to_wandb(mocker, monkeypatch, tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        wandb_entity="entity",
        wandb_authorized=True,
        wandb_approved_data_classes=("metrics", "provenance"),
    )
    run_spec = spec.initial_runs()[0]
    run = RunState(run_spec, status="completed", wandb_run_id="wandb-123")
    state = StudyState(spec.id, "study.yaml", "now", "now", {run_spec.id: run})
    remote_run = SimpleNamespace(summary={}, update=mocker.Mock())
    api = mocker.Mock()
    api.run.return_value = remote_run
    mocker.patch("wandb.Api", return_value=api)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    errors = publish_summaries_to_wandb(
        spec,
        state,
        {run_spec.id: make_summary(peak=0.8, time_to_95=100, time_auc=0.5)},
        {"runs": {}},
    )

    assert errors == []
    assert remote_run.summary["probe/peak_validation_accuracy"] == 0.8
    assert remote_run.summary["convergence/active_seconds_to_95"] == 100
    remote_run.update.assert_called_once_with()


def test_summary_refuses_unauthorized_external_publication(tmp_path: Path) -> None:
    spec = replace(make_spec(tmp_path), wandb_entity="entity")
    run_spec = spec.initial_runs()[0]
    run = RunState(run_spec, status="completed", wandb_run_id="wandb-123")
    state = StudyState(spec.id, "study.yaml", "now", "now", {run_spec.id: run})

    errors = publish_summaries_to_wandb(
        spec,
        state,
        {run_spec.id: make_summary(peak=0.8, time_to_95=100, time_auc=0.5)},
        {"runs": {}},
    )

    assert errors == ["W&B publication refused: study does not record explicit authorization"]


def test_summary_refuses_partial_external_publication(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        wandb_entity="entity",
        wandb_authorized=True,
        wandb_approved_data_classes=("metrics",),
    )
    run_spec = spec.initial_runs()[0]
    run = RunState(run_spec, status="completed", wandb_run_id="wandb-123")
    state = StudyState(spec.id, "study.yaml", "now", "now", {run_spec.id: run})

    errors = publish_summaries_to_wandb(
        spec,
        state,
        {run_spec.id: make_summary(peak=0.8, time_to_95=100, time_auc=0.5)},
        {"runs": {}},
    )

    assert errors == ["W&B publication refused: approval is missing for emitted data classes: provenance"]


def test_summary_refuses_an_implicit_emitted_data_manifest(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        wandb_entity="entity",
        wandb_authorized=True,
        wandb_approved_data_classes=("metrics", "provenance"),
        wandb_manifests_explicit=False,
    )
    run_spec = spec.initial_runs()[0]
    state = StudyState(
        spec.id,
        "study.yaml",
        "now",
        "now",
        {run_spec.id: RunState(run_spec, status="completed", wandb_run_id="wandb-123")},
    )

    errors = publish_summaries_to_wandb(spec, state, {}, {"runs": {}})

    assert errors == ["W&B publication refused: emitted-data manifest is not explicit"]


def test_online_wandb_requires_destination_authorization_and_every_emitted_class(tmp_path: Path) -> None:
    missing_destination = replace(make_spec(tmp_path), wandb_entity=None)
    partial_approval = replace(
        make_spec(tmp_path),
        wandb_entity="entity",
        wandb_authorized=True,
        wandb_approved_data_classes=("metrics",),
    )
    fully_approved = replace(
        partial_approval,
        wandb_approved_data_classes=("metrics", "configs", "provenance"),
    )

    assert any("destination" in error for error in wandb_preflight_errors(missing_destination, {}))
    assert any("configs" in error and "provenance" in error for error in wandb_preflight_errors(partial_approval, {}))
    assert wandb_preflight_errors(fully_approved, {}) == []
    assert wandb_preflight_errors(missing_destination, {"WANDB_MODE": "offline"}) == []


def test_authorized_wandb_study_refuses_local_mode(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        wandb_entity="entity",
        wandb_authorized=True,
        wandb_approved_data_classes=("metrics", "configs", "provenance"),
    )

    assert wandb_preflight_errors(spec, {"WANDB_MODE": "offline"}) == ["authorized W&B study requires online tracking"]


def test_wandb_gates_each_operation_against_its_emission_manifest(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        wandb_entity="entity",
        wandb_authorized=True,
        wandb_approved_data_classes=("metrics", "provenance"),
    )

    launch = spec.wandb_operation_decision("launch", "online")
    summary = spec.wandb_operation_decision("summary", "online")

    assert launch.effective_mode == "local-only"
    assert launch.missing_data_classes == ("configs",)
    assert summary.effective_mode == "online"
    assert summary.emitted_data_classes == ("metrics", "provenance")
    assert summary.to_dict()["destination"] == "entity/mjepa-cifar10"


def test_committed_studies_record_the_code_verified_wandb_manifests() -> None:
    expected = {key: tuple(sorted(value)) for key, value in WANDB_OPERATION_EMITTED_DATA_CLASSES.items()}

    for path in Path("research/studies").glob("*.yaml"):
        spec = StudySpec.from_path(path)
        assert spec.wandb_emitted_data_classes == expected
        assert spec.wandb_manifests_explicit


def test_register_root_cli_records_exact_root(tmp_path: Path, capsys) -> None:
    root = tmp_path / "managed" / "logs"

    assert research_main(["register-root", "--root", str(root)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["root"] == str(root.resolve())
    assert payload["marker"] == str(root / MANAGED_ROOT_MARKER_FILENAME)


def test_research_log_records_provenance_metrics_and_checkpoint_disposition(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, variants=0)
    run_spec = spec.initial_runs()[0]
    run_dir = spec.log_root / spec.id / "runs" / run_spec.id
    run_dir.mkdir(parents=True)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "parent": {"sha": "parent-sha", "branch": "codex/research/test-study"},
                "mjepa": {"sha": "mjepa-sha", "branch": "codex/research/test-study"},
                "vit": {"sha": "vit-sha", "branch": "master"},
            }
        )
    )
    state = StudyState(
        spec.id,
        "study.yaml",
        "now",
        "now",
        {
            run_spec.id: RunState(
                run_spec,
                status="completed",
                decision="baseline",
                run_dir=str(run_dir),
                checkpoint_disposition="retained",
                wandb_run_id="offline-id",
            )
        },
    )
    StateStore(spec.log_root / spec.id).save(state)
    summary = {
        "phase": "no-promotion",
        "winner": None,
        "pretraining": {run_spec.id: make_summary(peak=0.8, time_to_95=100, time_auc=0.5).to_dict()},
        "sft": {"runs": {}},
        "runs": {
            run_spec.id: {
                "status": "completed",
                "decision": "baseline",
                "wandb_url": "https://wandb.ai/entity/project/runs/id",
                "checkpoint_disposition": "retained",
            }
        },
    }

    assert append_research_log(spec, summary, tmp_path)

    research_log = (tmp_path / "research" / "LOG.md").read_text()
    assert "  - `baseline`: Mechanism: not recorded. Changes: not recorded." in research_log
    assert "parent=`parent-sha` (`codex/research/test-study`)" in research_log
    assert "peak_accuracy=0.800000" in research_log
    assert "active_seconds_to_95=100.000" in research_log
    assert "checkpoint=retained" in research_log
    assert "[run](https://wandb.ai/entity/project/runs/id)" in research_log


def test_research_log_describes_exploration_as_active(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, variants=1)
    StateStore(spec.log_root / spec.id).save(
        StudyState(
            spec.id,
            "study.yaml",
            "now",
            "now",
            {run_spec.id: RunState(run_spec) for run_spec in spec.initial_runs()},
            phase="exploration",
        )
    )
    summary = {
        "phase": "exploration",
        "winner": None,
        "pretraining": {},
        "sft": {"runs": {}},
        "runs": {},
    }

    assert append_research_log(spec, summary, tmp_path)

    research_log = (tmp_path / "research" / "LOG.md").read_text()
    assert (
        "Conclusion: No initial seed-0 candidate met a promotion threshold; "
        "bounded seed-0 exploration is still running."
    ) in research_log
    assert "Follow-up: run the preregistered exploration trials." in research_log


def test_research_log_records_each_retry_attempt_in_the_same_phase(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, variants=0)
    run_spec = spec.initial_runs()[0]
    run_dir = spec.log_root / spec.id / "runs" / run_spec.id
    run_dir.mkdir(parents=True)
    StateStore(spec.log_root / spec.id).save(
        StudyState(
            spec.id,
            "study.yaml",
            "now",
            "now",
            {run_spec.id: RunState(run_spec, status="completed", run_dir=str(run_dir))},
        )
    )

    def summary(attempt: int, event_id: str) -> dict[str, object]:
        return {
            "phase": "screening",
            "winner": None,
            "pretraining": {},
            "sft": {"runs": {}},
            "runs": {
                run_spec.id: {
                    "attempt": attempt,
                    "terminal_event_id": event_id,
                    "status": "completed",
                    "decision": "baseline",
                    "wandb_url": None,
                    "checkpoint_disposition": "retained",
                }
            },
        }

    assert append_research_log(spec, summary(1, "event-1"), tmp_path)
    assert append_research_log(spec, summary(2, "event-2"), tmp_path)

    research_log = (tmp_path / "research" / "LOG.md").read_text()
    assert research_log.count(f"`{run_spec.id}`: attempt=") == 2
    assert "attempt=1" in research_log
    assert "attempt=2" in research_log
