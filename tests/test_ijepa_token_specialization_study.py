import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "research" / "studies" / "ijepa-token-specialization-v1-screen.yaml"
SMOKE_STUDY_PATH = REPO_ROOT / "research" / "studies" / "ijepa-token-specialization-v1-smoke.yaml"
DIAGNOSTIC_PATH = REPO_ROOT / "research" / "diagnostics" / "ijepa-token-specialization-v1-screen.yaml"
RESULT_PATH = REPO_ROOT / "research" / "diagnostics" / "ijepa-token-specialization-v1-result-v1.json"
EXPECTED_VIT_SHA = "6d298af42b88f5d734554df28fef0198d6b0aed3"
CONTROL_LAUNCH_VIT_SHA = "f3b114c391414b2c2a4a0f4e04d7e9cadf9301ec"


def test_screen_is_a_fresh_paired_ijepa_comparison() -> None:
    study = yaml.safe_load(STUDY_PATH.read_text())

    assert study["baseline"]["id"] == "ijepa-control"
    assert study["baseline"]["config"] == "config/pretrain/vit-small.yaml"
    assert [(variant["id"], variant["config"]) for variant in study["variants"]] == [
        ("token-specialized", "config/pretrain/vit-small-ijepa-token-specialization.yaml")
    ]
    assert study["seeds"] == [0]
    assert study["resources"]["max_pretraining_trials"] == 2
    assert study["resources"]["max_concurrent_jobs"] == 2
    assert study["methodology"]["controls"]["official_test_set"] == "prohibited"
    assert not study["promotion"]["confirmation_enabled"]
    assert study["evaluation"]["official_test_roles"] == []
    assert study["code_shas"]["vit"] == EXPECTED_VIT_SHA
    assert CONTROL_LAUNCH_VIT_SHA in study["methodology"]["mechanical_retry"]["paired_control"]


def test_smoke_exercises_only_the_specialized_candidate() -> None:
    study = yaml.safe_load(SMOKE_STUDY_PATH.read_text())

    assert study["baseline"]["id"] == "token-specialized-smoke"
    assert study["variants"] == []
    assert study["resources"]["max_pretraining_trials"] == 1
    assert study["methodology"]["purpose"].startswith("Mechanical validation only")
    assert study["evaluation"]["official_test_roles"] == []


def test_diagnostics_compare_the_two_retained_backbones_without_test_data() -> None:
    manifest = yaml.safe_load(DIAGNOSTIC_PATH.read_text())

    assert [(source["id"], source["role"]) for source in manifest["sources"]] == [
        ("ijepa-control-seed0", "control"),
        ("token-specialized-seed0", "candidate"),
    ]
    assert manifest["data"]["split"] == "fixed-45000-train-5000-validation"
    assert manifest["data"]["official_test_set"] == "prohibited"
    assert manifest["diagnostics"]["gradient_mode"] == "torch.inference_mode"
    assert manifest["decision"]["candidate_to_control_centered_patch_energy_ratio_minimum"] == 0.90
    assert manifest["code_shas"]["vit"] == EXPECTED_VIT_SHA


def test_closeout_records_the_preregistered_mechanism_decision() -> None:
    study = yaml.safe_load(STUDY_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    gate = result["mechanism_support_gate"]
    observed = gate["observed"]
    thresholds = gate["thresholds"]

    assert result["status"] == "completed"
    assert result["decision"] == "mechanism-changed-but-support-gate-not-passed"
    assert result["smoke_validation"]["status"] == "completed"
    assert result["data"]["official_test_set_used"] is False
    assert result["scope"]["confirmation_run_launched"] is False
    assert gate["passed"] is False
    assert gate["failure_reasons"] == ["final-patch-mean-centroid-accuracy-gain"]
    assert thresholds == study["methodology"]["mechanism_support_gate"]
    assert observed["online_peak_accuracy_loss"]["passed"] is True
    assert observed["final_cls_centroid_accuracy_loss"]["passed"] is True
    assert observed["final_patch_mean_centroid_accuracy_gain"]["passed"] is False
    assert observed["final_cls_patch_alignment_decrease"]["passed"] is True
    assert observed["final_patch_pairwise_cosine_decrease"]["passed"] is True
    assert observed["candidate_to_control_centered_patch_energy_ratio"]["passed"] is True
    assert all(run["safety_gate_passed"] for run in result["runs"].values())
    assert all(run["checkpoint_disposition"] == "retained" for run in result["runs"].values())
