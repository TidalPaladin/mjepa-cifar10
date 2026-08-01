import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "research" / "probe-calibrations" / "lejepa-convergence-v1-probe.yaml"
OPTIMIZER_MANIFEST_PATH = REPO_ROOT / "research" / "probe-calibrations" / "lejepa-convergence-v1-optimizer-probe.yaml"


def load_calibration_script_module():
    module_path = REPO_ROOT / "scripts" / "calibrate_probes.py"
    spec = importlib.util.spec_from_file_location("probe_calibration_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_probe_calibration_manifest_is_complete_and_avoids_test_set() -> None:
    script = load_calibration_script_module()

    manifest, manifest_hash = script._load_manifest(MANIFEST_PATH)

    assert len(manifest["sources"]) == 7
    assert {recipe["mode"] for recipe in manifest["recipes"]} == {"final_cls", "last_two_cls"}
    assert manifest["data"]["official_test_set"] == "prohibited"
    assert len(manifest_hash) == 64
    for source in manifest["sources"]:
        run_dir = REPO_ROOT / source["run_dir"]
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / "backbone.safetensors").is_file()


def test_probe_calibration_manifest_requires_explicit_online_emissions() -> None:
    script = load_calibration_script_module()
    manifest, _ = script._load_manifest(MANIFEST_PATH)
    invalid_manifest = deepcopy(manifest)
    invalid_manifest["wandb"]["emitted_data_classes"]["launch"] = ["metrics"]

    with pytest.raises(ValueError, match="configs, metrics, and provenance"):
        script._validate_manifest(invalid_manifest)


def test_optimizer_probe_manifest_reuses_fixed_recipe_and_retained_sources() -> None:
    script = load_calibration_script_module()

    manifest, manifest_hash = script._load_manifest(OPTIMIZER_MANIFEST_PATH)

    assert [recipe["id"] for recipe in manifest["recipes"]] == ["last-two-cls-layernorm"]
    assert len(manifest["sources"]) == 5
    assert sum(source["role"] == "teacher-baseline" for source in manifest["sources"]) == 1
    assert manifest["data"]["official_test_set"] == "prohibited"
    assert len(manifest_hash) == 64
    for source in manifest["sources"]:
        run_dir = REPO_ROOT / source["run_dir"]
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / "backbone.safetensors").is_file()


def test_probe_curve_logging_uses_monotonic_global_steps(mocker) -> None:
    script = load_calibration_script_module()
    log = mocker.patch.object(script.wandb, "log")
    result = SimpleNamespace(validation_curves=((0.2, 0.3),))

    script._log_validation_curves("first", result, (0.01,), step_offset=0)
    script._log_validation_curves("second", result, (0.01,), step_offset=2)

    assert [call.kwargs["step"] for call in log.call_args_list] == [0, 1, 2, 3]
