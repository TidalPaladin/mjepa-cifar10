import json
import subprocess
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjepa_cifar10.research.inventory import index_local_runs, inventory_counts, open_inventory
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
    ResourceLimits,
    RunSpec,
    RunState,
    StudySpec,
    StudyState,
    VariantSpec,
)
from mjepa_cifar10.research.provenance import assert_launch_provenance
from mjepa_cifar10.research.runtime import (
    GIB,
    GPULock,
    StateStore,
    available_physical_gpus,
    build_worker_environment,
    cleanup_run_weights,
    estimate_checkpoint_size,
    prepare_retryable_runs,
    reconcile_state,
    required_free_bytes,
    run_command_with_timeout,
)
from mjepa_cifar10.research.summary import advance_study, append_research_log, publish_summaries_to_wandb


def make_summary(
    *,
    peak: float,
    time_to_95: float | None,
    time_auc: float,
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


def test_gpu_lock_excludes_duplicate_scheduler_assignment(tmp_path: Path) -> None:
    with GPULock(1, tmp_path):
        assert available_physical_gpus((1, 2), lock_root=tmp_path) == (2,)


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


def test_worker_environment_drops_inherited_wandb_service_socket(tmp_path: Path) -> None:
    environment = build_worker_environment(
        {"WANDB_SERVICE": "dead-socket-token", "WANDB_MODE": "offline"},
        physical_gpu=1,
        repo_root=tmp_path,
    )

    assert "WANDB_SERVICE" not in environment
    assert environment["WANDB_MODE"] == "offline"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


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
    retryable.pid = 123
    retryable_run_dir = tmp_path / "runs" / retryable.spec.id
    retryable_run_dir.mkdir(parents=True)
    retryable.run_dir = str(retryable_run_dir)
    (retryable_run_dir / "terminal.json").write_text('{"status": "failed"}')
    retained = list(runs.values())[1]
    retained.status = "completed"
    retained.decision = "rejected"
    state = StudyState(spec.id, "study.yaml", "now", "now", runs)

    assert prepare_retryable_runs(state) == 1

    assert retryable.status == "pending"
    assert retryable.decision == "pending"
    assert retryable.pid is None
    assert not (retryable_run_dir / "terminal.json").exists()
    assert len(tuple((retryable_run_dir / "attempts").glob("terminal-*.json"))) == 1
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


def test_summary_publishes_standardized_fields_to_wandb(mocker, monkeypatch, tmp_path: Path) -> None:
    spec = replace(make_spec(tmp_path), wandb_entity="entity")
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
