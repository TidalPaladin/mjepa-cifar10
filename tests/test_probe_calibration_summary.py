import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_summary_script_module():
    module_path = REPO_ROOT / "scripts" / "summarize_probe_calibration.py"
    spec = importlib.util.spec_from_file_location("probe_calibration_summary_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_result(source_id: str, role: str, online: float, calibrated: float) -> dict[str, object]:
    return {
        "status": "completed",
        "source_id": source_id,
        "source_role": role,
        "manifest_sha256": "manifest-hash",
        "online_probe_accuracy": online,
        "best_calibrated_accuracy": calibrated,
        "calibration_gain": calibrated - online,
        "best_recipe": "last-two-cls-layernorm",
        "recipes": {},
        "active_seconds": 1.0,
        "wandb_run_id": f"wandb-{source_id}",
        "wandb_url": f"https://wandb.invalid/{source_id}",
    }


def test_summary_declares_representation_primary_after_material_probe_gain() -> None:
    script = load_summary_script_module()
    manifest = {
        "id": "calibration",
        "sources": [
            {"id": "teacher", "role": "teacher-baseline"},
            {"id": "shared", "role": "shared-student-candidate"},
        ],
        "decision": {
            "material_probe_lag_gain": 0.10,
            "shared_representation_floor": 0.60,
            "maximum_teacher_gap_for_probe_explanation": 0.20,
        },
    }
    results = {
        "teacher": make_result("teacher", "teacher-baseline", 0.90, 0.90),
        "shared": make_result("shared", "shared-student-candidate", 0.40, 0.58),
    }

    summary = script._build_summary(manifest, "manifest-hash", results)

    assert summary["best_shared_source"] == "shared"
    assert summary["material_probe_lag"] is True
    assert summary["teacher_gap"] == pytest.approx(0.32)
    assert summary["representation_convergence_primary"] is True
    assert summary["decision"] == "material-probe-lag-with-residual-representation-gap"


def test_summary_rejects_missing_source_results() -> None:
    script = load_summary_script_module()
    manifest = {
        "id": "calibration",
        "sources": [{"id": "teacher", "role": "teacher-baseline"}],
        "decision": {},
    }

    with pytest.raises(ValueError, match="result source IDs do not match"):
        script._build_summary(manifest, "manifest-hash", {})


def test_summary_compares_control_and_candidate_sources() -> None:
    script = load_summary_script_module()
    manifest = {
        "id": "control-candidate-calibration",
        "sources": [
            {"id": "control", "role": "control"},
            {"id": "candidate", "role": "candidate"},
        ],
        "decision": {"selection": "Apply preregistered gates in the parent study."},
    }
    results = {
        "control": make_result("control", "control", 0.39, 0.50),
        "candidate": make_result("candidate", "candidate", 0.48, 0.60),
    }

    summary = script._build_summary(manifest, "manifest-hash", results)

    assert summary["control_source"] == "control"
    assert summary["best_candidate_source"] == "candidate"
    assert summary["control_calibrated_accuracy"] == pytest.approx(0.50)
    assert summary["best_candidate_calibrated_accuracy"] == pytest.approx(0.60)
    assert summary["calibrated_accuracy_gain"] == pytest.approx(0.10)
    assert summary["online_accuracy_gain"] == pytest.approx(0.09)
    assert summary["decision"] == "candidate-higher-calibrated-accuracy"
