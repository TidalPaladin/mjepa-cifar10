from dataclasses import fields
from pathlib import Path

import yaml
from mjepa import JEPAConfig, OptimizerConfig, TrainerConfig  # noqa: F401
from vit import ViTConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG_PATH = REPO_ROOT / "config" / "pretrain" / "vit-small-ijepa-token-specialization.yaml"
CANDIDATE_CONFIG_PATH = REPO_ROOT / "config" / "pretrain" / "vit-small-ijepa-token-specialization-no-registers.yaml"
STUDY_PATH = REPO_ROOT / "research" / "studies" / "ijepa-token-specialization-v2-no-registers.yaml"
DIAGNOSTIC_PATH = REPO_ROOT / "research" / "diagnostics" / "ijepa-token-specialization-v2-no-registers.yaml"
EXPECTED_VIT_SHA = "859b7ac772ba16d9febd3c6d746de818d2cc11fb"


def test_candidate_changes_only_the_register_token_count() -> None:
    reference = yaml.full_load(REFERENCE_CONFIG_PATH.read_text())
    candidate = yaml.full_load(CANDIDATE_CONFIG_PATH.read_text())
    reference_backbone = reference["backbone"]
    candidate_backbone = candidate["backbone"]

    assert isinstance(reference_backbone, ViTConfig)
    assert isinstance(candidate_backbone, ViTConfig)
    assert reference_backbone.num_register_tokens == 7
    assert candidate_backbone.num_register_tokens == 0
    assert candidate_backbone.num_cls_tokens == 1
    assert candidate_backbone.specialize_global_token_norms
    assert candidate_backbone.specialize_global_token_qkv_blocks == 4
    for field in fields(ViTConfig):
        if field.name != "num_register_tokens":
            assert getattr(candidate_backbone, field.name) == getattr(reference_backbone, field.name)
    for section in ("trainer", "jepa", "optimizer"):
        assert candidate[section] == reference[section]


def test_study_is_a_fresh_paired_register_ablation() -> None:
    study = yaml.safe_load(STUDY_PATH.read_text())

    assert study["baseline"]["id"] == "token-specialized-register7"
    assert study["baseline"]["config"] == "config/pretrain/vit-small-ijepa-token-specialization.yaml"
    assert [(variant["id"], variant["config"]) for variant in study["variants"]] == [
        (
            "token-specialized-register0",
            "config/pretrain/vit-small-ijepa-token-specialization-no-registers.yaml",
        )
    ]
    assert "baseline_reference" not in study
    assert study["seeds"] == [0]
    assert study["resources"]["max_pretraining_trials"] == 2
    assert study["resources"]["max_concurrent_jobs"] == 2
    assert not study["promotion"]["confirmation_enabled"]
    assert study["evaluation"]["official_test_roles"] == []
    assert study["code_shas"]["vit"] == EXPECTED_VIT_SHA


def test_diagnostics_compare_register_counts_without_test_data() -> None:
    manifest = yaml.safe_load(DIAGNOSTIC_PATH.read_text())

    assert [(source["id"], source["role"]) for source in manifest["sources"]] == [
        ("token-specialized-register7-seed0", "control"),
        ("token-specialized-register0-seed0", "candidate"),
    ]
    assert manifest["data"]["split"] == "fixed-45000-train-5000-validation"
    assert manifest["data"]["official_test_set"] == "prohibited"
    assert manifest["diagnostics"]["gradient_mode"] == "torch.inference_mode"
    assert manifest["decision"]["candidate_to_control_centered_patch_energy_ratio_minimum"] == 0.90
    assert manifest["code_shas"]["vit"] == EXPECTED_VIT_SHA
