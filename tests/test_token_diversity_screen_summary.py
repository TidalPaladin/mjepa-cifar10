import importlib.util
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "research" / "probe-calibrations" / "lejepa-token-diversity-v1-objective-probe.yaml"


def load_summary_script_module():
    module_path = REPO_ROOT / "scripts" / "summarize_token_diversity_screen.py"
    spec = importlib.util.spec_from_file_location("token_diversity_screen_summary_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_record(
    *,
    step: int,
    accuracy: float,
    patch_cosine: float,
    patch_energy: float,
    rank: float = 0.10,
) -> dict[str, float | int]:
    return {
        "_step": step,
        "convergence/active_seconds": float(step) * 2.0,
        "probe/validation_accuracy": accuracy,
        "pretrain/collapse/target_cls/finite_fraction": 1.0,
        "pretrain/collapse/target_cls/std_mean": 0.20,
        "pretrain/collapse/target_cls/effective_rank_fraction": rank,
        "pretrain/collapse/target_cls/top_eigenvalue_fraction": 0.30,
        "pretrain/collapse/target_cls/mean_pairwise_cosine": 0.10,
        "pretrain/collapse/target_patch_mean/finite_fraction": 1.0,
        "pretrain/collapse/target_patch_mean/std_mean": 0.20,
        "pretrain/collapse/target_patch_mean/effective_rank_fraction": rank,
        "pretrain/collapse/target_patch_mean/top_eigenvalue_fraction": 0.30,
        "pretrain/collapse/target_patch_mean/mean_pairwise_cosine": 0.10,
        "pretrain/diversity/target_patch/finite_image_fraction": 1.0,
        "pretrain/diversity/target_patch/mean_within_image_pairwise_cosine": patch_cosine,
        "pretrain/diversity/target_patch/centered_patch_energy_ratio": patch_energy,
        "pretrain/diversity/target_patch/centered_patch_effective_rank_fraction": 0.10,
        "pretrain/validation/cpa_mean": 0.50,
        "pretrain/validation_visual_target_relative_improvement": 0.20,
    }


def make_probe_result(source_id: str, role: str, accuracy: float) -> dict[str, object]:
    return {
        "status": "completed",
        "source_id": source_id,
        "source_role": role,
        "manifest_sha256": "manifest-hash",
        "best_calibrated_accuracy": accuracy,
        "best_recipe": "last-two-cls-layernorm",
        "calibration_gain": 0.0,
    }


def make_screen_inputs(*, candidate_rank: float = 0.10):
    study = {
        "id": "screen",
        "methodology": {
            "last_three_validation_gate": {
                "target_cls_std_mean_minimum": 0.10,
                "target_cls_effective_rank_fraction_minimum": 0.07,
                "target_cls_top_eigenvalue_fraction_maximum": 0.50,
                "target_cls_mean_pairwise_cosine_maximum": 0.90,
                "target_patch_mean_std_mean_minimum": 0.10,
                "target_patch_mean_effective_rank_fraction_minimum": 0.07,
                "target_patch_mean_top_eigenvalue_fraction_maximum": 0.50,
                "target_patch_mean_mean_pairwise_cosine_maximum": 0.90,
                "visual_target_relative_improvement_minimum": 0.01,
                "finite_fraction_required": 1.0,
            },
            "terminal_selection_gate": {
                "candidate_to_control_centered_patch_energy_ratio_minimum": 1.50,
                "candidate_patch_pair_cosine_improvement_minimum": 0.05,
                "frozen_accuracy_gain_minimum": 0.02,
                "frozen_equivalence_accuracy_loss_maximum": 0.005,
                "online_step_auc_gain_if_frozen_equivalent": 0.10,
            },
        },
    }
    manifest = {
        "id": "probe",
        "sources": [
            {"id": "control", "role": "control", "objective_complexity_rank": 0},
            {"id": "candidate", "role": "candidate", "objective_complexity_rank": 1},
        ],
    }
    records = {
        "control": [
            make_record(step=step, accuracy=accuracy, patch_cosine=0.90, patch_energy=0.10)
            for step, accuracy in ((1, 0.20), (2, 0.25), (3, 0.30))
        ],
        "candidate": [
            make_record(
                step=step,
                accuracy=accuracy,
                patch_cosine=0.70,
                patch_energy=0.30,
                rank=candidate_rank,
            )
            for step, accuracy in ((1, 0.23), (2, 0.29), (3, 0.34))
        ],
    }
    results = {
        "control": make_probe_result("control", "control", 0.50),
        "candidate": make_probe_result("candidate", "candidate", 0.53),
    }
    return study, manifest, records, results


def test_committed_objective_probe_manifest_covers_only_the_four_preregistered_encoders() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())

    assert manifest["screen_study"] == "research/studies/lejepa-token-diversity-v1-objective-screen.yaml"
    assert [source["role"] for source in manifest["sources"]].count("control") == 1
    assert [source["role"] for source in manifest["sources"]].count("candidate") == 3
    assert [recipe["id"] for recipe in manifest["recipes"]] == ["last-two-cls-layernorm"]
    assert manifest["probe"]["epochs"] == 100
    assert len(manifest["probe"]["learning_rates"]) == 6
    assert manifest["data"]["official_test_set"] == "prohibited"
    for source in manifest["sources"]:
        run_dir = REPO_ROOT / source["run_dir"]
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / "backbone.safetensors").is_file()


def test_summary_selects_at_most_one_candidate_that_passes_every_gate() -> None:
    script = load_summary_script_module()
    study, manifest, records, results = make_screen_inputs()

    summary = script._build_summary(study, manifest, "manifest-hash", records, results)

    assert summary["selected_source"] == "candidate"
    assert summary["decision"] == "candidate-selected-for-preregistered-long-horizon"
    assert summary["runs"]["candidate"]["qualifies"]
    assert summary["runs"]["candidate"]["spatial_gate_passed"]
    assert summary["runs"]["candidate"]["frozen_gate_passed"]
    assert len(summary["qualifying_sources"]) == 1


def test_summary_rejects_spatial_and_frozen_winner_when_collapse_gate_fails() -> None:
    script = load_summary_script_module()
    study, manifest, records, results = make_screen_inputs(candidate_rank=0.05)

    summary = script._build_summary(study, manifest, "manifest-hash", records, results)

    assert summary["selected_source"] is None
    assert summary["decision"] == "no-candidate-passed-preregistered-gates"
    assert not summary["runs"]["candidate"]["collapse_gate_passed"]
    assert "collapse-gate" in summary["runs"]["candidate"]["failure_reasons"]
