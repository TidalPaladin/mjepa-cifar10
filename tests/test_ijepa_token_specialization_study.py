from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "research" / "studies" / "ijepa-token-specialization-v1-screen.yaml"
SMOKE_STUDY_PATH = REPO_ROOT / "research" / "studies" / "ijepa-token-specialization-v1-smoke.yaml"
DIAGNOSTIC_PATH = REPO_ROOT / "research" / "diagnostics" / "ijepa-token-specialization-v1-screen.yaml"
EXPECTED_VIT_SHA = "f3b114c391414b2c2a4a0f4e04d7e9cadf9301ec"


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
