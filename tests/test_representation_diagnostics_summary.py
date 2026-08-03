import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script_module():
    module_path = REPO_ROOT / "scripts" / "summarize_representation_diagnostics.py"
    spec = importlib.util.spec_from_file_location("representation_diagnostics_summary_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(role: str, cls_accuracy: float, patch_accuracy: float, cpa: float, patch_cosine: float, energy: float):
    return {
        "status": "completed",
        "manifest_sha256": "manifest-hash",
        "source_role": role,
        "layers": [
            {
                "layer": 1,
                "centroid_accuracy": {
                    "cls": cls_accuracy,
                    "patch_mean": patch_accuracy,
                    "cls_patch_mean": max(cls_accuracy, patch_accuracy),
                },
                "cls_patch_alignment": {"cpa_mean": cpa},
                "patch_diversity": {
                    "mean_within_image_pairwise_cosine": patch_cosine,
                    "centered_patch_energy_ratio": energy,
                },
            }
        ],
    }


def manifest():
    return {
        "id": "diagnostic",
        "sources": [
            {"id": "teacher", "role": "teacher-baseline"},
            {"id": "candidate", "role": "shared-student-candidate"},
        ],
        "decision": {
            "material_centroid_accuracy_gap": 0.05,
            "minimum_candidate_cpa_excess": 0.20,
            "minimum_candidate_patch_pair_cosine_excess": 0.20,
            "maximum_candidate_to_teacher_centered_energy_ratio": 0.75,
        },
    }


def test_summary_supports_spatial_homogenization_when_every_gate_passes() -> None:
    script = load_script_module()
    results = {
        "teacher": result("teacher-baseline", 0.9, 0.85, 0.4, 0.3, 0.8),
        "candidate": result("shared-student-candidate", 0.8, 0.7, 0.9, 0.8, 0.4),
    }

    summary = script._build_summary(manifest(), "manifest-hash", results)

    assert summary["spatial_homogenization_supported"] is True
    assert summary["representation_gap_supported"] is True
    assert summary["decision"] == "spatial-homogenization-with-representation-gap"
    assert summary["first_material_cls_gap_layer"] == 1


def test_summary_rejects_homogenization_when_centered_energy_is_preserved() -> None:
    script = load_script_module()
    results = {
        "teacher": result("teacher-baseline", 0.9, 0.85, 0.4, 0.3, 0.8),
        "candidate": result("shared-student-candidate", 0.8, 0.7, 0.9, 0.8, 0.7),
    }

    summary = script._build_summary(manifest(), "manifest-hash", results)

    assert summary["spatial_homogenization_supported"] is False
    assert summary["decision"] == "representation-gap-without-spatial-homogenization"
