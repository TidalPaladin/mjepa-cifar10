import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "research" / "probe-calibrations" / "lejepa-convergence-v1-probe.yaml"


def load_worker_script_module():
    module_path = REPO_ROOT / "scripts" / "run_probe_calibration_workers.py"
    spec = importlib.util.spec_from_file_location("probe_calibration_workers_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_commands_follow_manifest_gpu_assignment() -> None:
    script = load_worker_script_module()

    commands = script._build_worker_commands(MANIFEST_PATH, Path("/mnt/data/cifar10"), [1, 2])

    assert len(commands) == 2
    for worker_index, (command, environment) in enumerate(commands):
        assert command[0] == sys.executable
        assert command[-8:] == [
            "--worker-index",
            str(worker_index),
            "--num-workers",
            "2",
            "--local-rank",
            "0",
            "--physical-gpu",
            str(worker_index + 1),
        ]
        assert environment["CUDA_VISIBLE_DEVICES"] == str(worker_index + 1)
