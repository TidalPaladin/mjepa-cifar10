import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "research" / "diagnostics" / "lejepa-token-diversity-v1-audit.yaml"


def load_script_module():
    module_path = REPO_ROOT / "scripts" / "diagnose_representations.py"
    spec = importlib.util.spec_from_file_location("representation_diagnostics_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_representation_diagnostic_manifest_is_safe_and_complete() -> None:
    script = load_script_module()

    manifest, manifest_hash = script._load_manifest(MANIFEST_PATH)

    assert manifest_hash
    assert manifest["data"]["split"] == "fixed-45000-train-5000-validation"
    assert manifest["data"]["official_test_set"] == "prohibited"
    assert [source["role"] for source in manifest["sources"]] == [
        "teacher-baseline",
        "shared-student-candidate",
    ]
    assert manifest["wandb"]["emitted_data_classes"]["launch"] == ["metrics", "configs", "provenance"]


def test_probe_feature_dimensions_keep_routes_separate() -> None:
    script = load_script_module()

    dimensions = script._probe_feature_dims(384)

    assert dimensions == {"cls": 384, "patch_mean": 384, "cls_patch_mean": 768}
