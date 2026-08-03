import importlib.util
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_MANIFEST_PATH = REPO_ROOT / "research" / "probe-calibrations" / "lejepa-patch-rank-v1-probe.yaml"
DIAGNOSTIC_MANIFEST_PATH = REPO_ROOT / "research" / "diagnostics" / "lejepa-patch-rank-v1-screen.yaml"


def load_summary_script_module():
    module_path = REPO_ROOT / "scripts" / "summarize_patch_rank_screen.py"
    spec = importlib.util.spec_from_file_location("patch_rank_screen_summary_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_record(
    *,
    step: int,
    accuracy: float,
    patch_mean_rank: float,
    patch_cosine: float,
    patch_energy: float,
) -> dict[str, float | int]:
    return {
        "_step": step,
        "convergence/active_seconds": float(step) * 2.0,
        "probe/validation_accuracy": accuracy,
        "pretrain/collapse/target_cls/finite_fraction": 1.0,
        "pretrain/collapse/target_cls/std_mean": 0.20,
        "pretrain/collapse/target_cls/top_eigenvalue_fraction": 0.30,
        "pretrain/collapse/target_cls/mean_pairwise_cosine": 0.10,
        "pretrain/collapse/target_patch_mean/finite_fraction": 1.0,
        "pretrain/collapse/target_patch_mean/std_mean": 0.20,
        "pretrain/collapse/target_patch_mean/effective_rank_fraction": patch_mean_rank,
        "pretrain/collapse/target_patch_mean/top_eigenvalue_fraction": 0.30,
        "pretrain/collapse/target_patch_mean/mean_pairwise_cosine": 0.10,
        "pretrain/diversity/target_patch/finite_image_fraction": 1.0,
        "pretrain/diversity/target_patch/mean_within_image_pairwise_cosine": patch_cosine,
        "pretrain/diversity/target_patch/centered_patch_energy_ratio": patch_energy,
        "pretrain/validation_visual_target_relative_improvement": 0.20,
    }


def make_probe_result(source_id: str, role: str, accuracy: float) -> dict[str, object]:
    return {
        "status": "completed",
        "source_id": source_id,
        "source_role": role,
        "manifest_sha256": "probe-hash",
        "best_calibrated_accuracy": accuracy,
        "best_recipe": "last-two-cls-layernorm",
        "calibration_gain": 0.0,
    }


def make_diagnostic_result(
    source_id: str,
    role: str,
    *,
    block8_patch_accuracy: float,
    final_patch_accuracy: float,
) -> dict[str, object]:
    layers = [
        {
            "layer": layer,
            "centroid_accuracy": {
                "cls": 0.40,
                "patch_mean": block8_patch_accuracy if layer == 8 else final_patch_accuracy if layer == 12 else 0.30,
                "cls_patch_mean": 0.45,
            },
        }
        for layer in range(1, 13)
    ]
    return {
        "status": "completed",
        "source_id": source_id,
        "source_role": role,
        "manifest_sha256": "diagnostic-hash",
        "layers": layers,
    }


def make_screen_inputs(*, candidate_frozen_accuracy: float = 0.499):
    study = {
        "id": "screen",
        "methodology": {
            "safety_gate": {
                "finite_fraction_required": 1.0,
                "target_cls_std_mean_minimum": 0.10,
                "target_cls_top_eigenvalue_fraction_maximum": 0.50,
                "target_cls_mean_pairwise_cosine_maximum": 0.90,
                "target_patch_mean_std_mean_minimum": 0.10,
                "target_patch_mean_top_eigenvalue_fraction_maximum": 0.50,
                "target_patch_mean_mean_pairwise_cosine_maximum": 0.90,
                "visual_target_relative_improvement_minimum": 0.01,
            },
            "terminal_selection_gate": {
                "last_three_patch_mean_rank_ratio_minimum": 1.50,
                "final_patch_centroid_accuracy_gain_minimum": 0.03,
                "candidate_to_control_centered_patch_energy_ratio_minimum": 0.80,
                "candidate_patch_cosine_maximum_increase": 0.05,
                "frozen_accuracy_loss_maximum": 0.005,
                "online_step_auc_loss_maximum": 0.01,
            },
        },
    }
    probe_manifest = {
        "id": "probe",
        "sources": [
            {"id": "control", "role": "control", "objective_complexity_rank": 0, "objective_cost_rank": 0},
            {"id": "candidate", "role": "candidate", "objective_complexity_rank": 1, "objective_cost_rank": 1},
        ],
    }
    diagnostic_manifest = {
        "id": "diagnostics",
        "sources": [
            {"id": "control", "role": "control"},
            {"id": "candidate", "role": "candidate"},
        ],
    }
    records = {
        "control": [
            make_record(
                step=step,
                accuracy=accuracy,
                patch_mean_rank=0.10,
                patch_cosine=0.40,
                patch_energy=0.50,
            )
            for step, accuracy in ((1, 0.30), (2, 0.40), (3, 0.50))
        ],
        "candidate": [
            make_record(
                step=step,
                accuracy=accuracy,
                patch_mean_rank=0.16,
                patch_cosine=0.44,
                patch_energy=0.45,
            )
            for step, accuracy in ((1, 0.29), (2, 0.39), (3, 0.49))
        ],
    }
    probe_results = {
        "control": make_probe_result("control", "control", 0.50),
        "candidate": make_probe_result("candidate", "candidate", candidate_frozen_accuracy),
    }
    diagnostic_results = {
        "control": make_diagnostic_result("control", "control", block8_patch_accuracy=0.30, final_patch_accuracy=0.35),
        "candidate": make_diagnostic_result(
            "candidate",
            "candidate",
            block8_patch_accuracy=0.31,
            final_patch_accuracy=0.39,
        ),
    }
    return study, probe_manifest, diagnostic_manifest, records, probe_results, diagnostic_results


def test_committed_manifests_cover_only_the_four_preregistered_encoders() -> None:
    probe_manifest = yaml.safe_load(PROBE_MANIFEST_PATH.read_text())
    diagnostic_manifest = yaml.safe_load(DIAGNOSTIC_MANIFEST_PATH.read_text())

    expected_ids = [
        "patch-residual-control-seed0",
        "patch-residual-l010-seed0",
        "patch-sample010-seed0",
        "patch-sample050-seed0",
    ]
    assert probe_manifest["screen_study"] == "research/studies/lejepa-patch-rank-v1-screen.yaml"
    assert probe_manifest["diagnostic_manifest"] == "research/diagnostics/lejepa-patch-rank-v1-screen.yaml"
    assert [source["id"] for source in probe_manifest["sources"]] == expected_ids
    assert [source["id"] for source in diagnostic_manifest["sources"]] == expected_ids
    assert [source["role"] for source in probe_manifest["sources"]].count("control") == 1
    assert [recipe["id"] for recipe in probe_manifest["recipes"]] == ["last-two-cls-layernorm"]
    assert probe_manifest["probe"]["epochs"] == 100
    assert len(probe_manifest["probe"]["learning_rates"]) == 6
    assert probe_manifest["data"]["official_test_set"] == "prohibited"
    assert diagnostic_manifest["data"]["official_test_set"] == "prohibited"
    for source in probe_manifest["sources"]:
        run_dir = REPO_ROOT / source["run_dir"]
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / "backbone.safetensors").is_file()


def test_summary_selects_at_most_one_candidate_that_passes_every_gate() -> None:
    script = load_summary_script_module()
    inputs = make_screen_inputs()

    summary = script._build_summary(*inputs[:3], "probe-hash", "diagnostic-hash", *inputs[3:])

    assert summary["selected_source"] == "candidate"
    assert summary["decision"] == "candidate-selected-for-preregistered-long-horizon"
    assert summary["runs"]["candidate"]["qualifies"]
    assert summary["runs"]["candidate"]["last_three_patch_mean_rank_ratio"] == pytest.approx(1.6)
    assert summary["runs"]["candidate"]["final_patch_centroid_accuracy_gain"] == pytest.approx(0.04)
    assert summary["runs"]["candidate"]["patch_centroid_block8_to_final_gain"] == pytest.approx(0.08)
    assert len(summary["qualifying_sources"]) == 1


def test_summary_rejects_candidate_when_frozen_accuracy_exceeds_loss_budget() -> None:
    script = load_summary_script_module()
    inputs = make_screen_inputs(candidate_frozen_accuracy=0.49)

    summary = script._build_summary(*inputs[:3], "probe-hash", "diagnostic-hash", *inputs[3:])

    assert summary["selected_source"] is None
    assert summary["decision"] == "no-candidate-passed-preregistered-gates"
    assert not summary["runs"]["candidate"]["frozen_accuracy_gate_passed"]
    assert "frozen-accuracy" in summary["runs"]["candidate"]["failure_reasons"]
