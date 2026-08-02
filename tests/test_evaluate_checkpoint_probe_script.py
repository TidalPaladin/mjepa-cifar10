import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "evaluate_checkpoint_probe.py"
TERMINAL_STEP = 17_418
TERMINAL_EPOCH = 396
TERMINAL_ACCURACY = 0.8104
TERMINAL_ACTIVE_SECONDS = 43_094.0


def load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_checkpoint_probe_script_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_grad_enabled: list[bool] = []

    def forward_target(self, images: Tensor) -> Tensor:
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        return images

    def forward_probe(self, features: Tensor) -> dict[str, Tensor]:
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        return {"cls": features}


def test_evaluate_probe_uses_eval_inference_and_counts_examples() -> None:
    module = load_script_module()
    logits = torch.zeros(3, module.NUM_CLASSES)
    logits[0, 0] = 3.0
    logits[1, 1] = 3.0
    logits[2, 1] = 3.0
    labels = torch.tensor([0, 1, 0])
    dataloader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    model = FakeProbeModel()
    model.train()

    result = module.evaluate_probe(model, dataloader, torch.device("cpu"), autocast_dtype=None)

    assert model.training is False
    assert model.forward_grad_enabled == [False, False, False, False]
    assert result.correct == 2
    assert result.total == 3
    assert result.accuracy == pytest.approx(2 / 3)


def test_checkpoint_metadata_rejects_missing_endpoint() -> None:
    module = load_script_module()

    with pytest.raises(ValueError, match="step and epoch"):
        module.checkpoint_endpoint({"backbone": {}})


def test_write_result_is_atomic_json(tmp_path: Path) -> None:
    module = load_script_module()
    output = tmp_path / "terminal-probe.json"

    module.write_result(output, {"accuracy": TERMINAL_ACCURACY, "step": TERMINAL_STEP})

    assert json.loads(output.read_text()) == {"accuracy": TERMINAL_ACCURACY, "step": TERMINAL_STEP}
    assert not tuple(tmp_path.glob("*.tmp"))


def test_append_endpoint_metric_is_idempotent_and_rejects_conflicts(tmp_path: Path) -> None:
    module = load_script_module()
    result = {
        "checkpoint": {
            "step": TERMINAL_STEP,
            "epoch": TERMINAL_EPOCH,
            "active_seconds": TERMINAL_ACTIVE_SECONDS,
        },
        "evaluation": {"accuracy": TERMINAL_ACCURACY},
    }

    assert module.append_endpoint_metric(tmp_path, result) is True
    assert module.append_endpoint_metric(tmp_path, result) is False
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1

    conflicting_result = {**result, "evaluation": {"accuracy": TERMINAL_ACCURACY - 0.01}}
    with pytest.raises(ValueError, match="conflicting terminal probe metric"):
        module.append_endpoint_metric(tmp_path, conflicting_result)
