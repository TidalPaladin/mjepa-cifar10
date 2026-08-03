import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

from mjepa_cifar10.research.codex_notifications import notification_path_for_event
from mjepa_cifar10.research.models import StudySpec
from mjepa_cifar10.research.runtime import StateStore, cleanup_run_weights, launch_available_runs, reconcile_state
from mjepa_cifar10.research.summary import summarize_study


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_TIMEOUT_SECONDS = 15 * 60
POLL_INTERVAL_SECONDS = 5
DATA_ENVIRONMENT_VARIABLE = "CIFAR10_DATA"


@pytest.mark.ci_skip
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="requires physical GPU 1 or 2",
)
def test_wandb_offline_managed_study_checkpoint_resume_summary_and_retention(tmp_path: Path, monkeypatch) -> None:
    dataset_path = os.environ.get(DATA_ENVIRONMENT_VARIABLE)
    if dataset_path is None:
        pytest.skip(f"{DATA_ENVIRONMENT_VARIABLE} is not set")
    study_path = tmp_path / "smoke.yaml"
    log_root = tmp_path / "logs"
    study_path.write_text(
        f"""
id: gpu-smoke
question: Can the managed GPU smoke run recover?
hypothesis: One epoch will produce every managed artifact.
baseline:
  id: baseline
  config: {REPO_ROOT / "config/pretrain/smoke.yaml"}
  hypothesis: Harness smoke baseline.
variants: []
data: {dataset_path}
log_root: {log_root}
seeds: [0, 1, 2]
wandb:
  entity: tidalpaladin
  project: mjepa-cifar10
  group: gpu-smoke
  emitted_data_classes:
    launch: [metrics, configs, provenance]
    summary: [metrics, provenance]
code_shas: {{}}
resources:
  physical_gpus: [1, 2]
  max_concurrent_jobs: 1
  timeout_seconds: {SMOKE_TIMEOUT_SECONDS}
  max_pretraining_trials: 8
  minimum_free_gib: 1
  fallback_checkpoint_gib: 1
"""
    )
    spec = StudySpec.from_path(study_path)
    monkeypatch.setenv("WANDB_MODE", "offline")
    lock_root = tmp_path / "locks"

    state = launch_available_runs(spec, study_path, REPO_ROOT, development=True, lock_root=lock_root)
    run = next(iter(state.runs.values()))
    deadline = time.monotonic() + SMOKE_TIMEOUT_SECONDS
    while run.status not in ("completed", "failed", "timed_out") and time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        with StateStore(log_root / spec.id) as store:
            state = store.load()
            reconcile_state(state)
            store.save(state)
        run = state.runs[run.spec.id]

    assert run.status == "completed", run.error
    assert run.run_dir is not None
    run_dir = Path(run.run_dir)
    checkpoint = run_dir / "checkpoint.pt"
    assert checkpoint.is_file()
    for artifact in (
        "config.yaml",
        "metrics.jsonl",
        "metadata.json",
        "provenance.json",
        "terminal.json",
    ):
        assert (run_dir / artifact).is_file()
    assert run.terminal_event_id is not None
    assert notification_path_for_event(log_root, run.terminal_event_id).is_file()
    assert not (run_dir / "notification.json").exists()
    tracker = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))["external_tracker"]
    assert tracker["operation"] == "launch"
    assert tracker["requested_mode"] == "offline"
    assert tracker["effective_mode"] == "local-only"
    assert tracker["emitted_data_classes"] == ["configs", "metrics", "provenance"]

    resume_environment = dict(os.environ)
    resume_environment["CUDA_VISIBLE_DEVICES"] = "2"
    subprocess.run(
        (
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "scripts/pretrain.py"),
            str(REPO_ROOT / "config/pretrain/smoke.yaml"),
            dataset_path,
            "--checkpoint",
            str(checkpoint),
            "--exact-log-dir",
            str(run_dir),
            "--local-rank",
            "0",
            "--wandb-run-id",
            run.wandb_run_id or "offline-smoke",
        ),
        cwd=REPO_ROOT,
        env=resume_environment,
        check=True,
        timeout=SMOKE_TIMEOUT_SECONDS,
    )

    summary = summarize_study(spec, study_path, REPO_ROOT)
    assert summary["phase"] == "no-promotion"
    assert summary["pretraining"]
    with StateStore(log_root / spec.id) as store:
        state = store.load()
    with pytest.raises(ValueError, match="decision 'baseline'"):
        cleanup_run_weights(state, run.spec.id, log_root / spec.id)
