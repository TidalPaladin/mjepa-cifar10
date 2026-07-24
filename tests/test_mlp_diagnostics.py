import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from mjepa import JEPAConfig
from mjepa.jepa import CrossAttentionPredictor
from torch import nn
from vit import ViTConfig
from vit.fused import NormMLP

import mjepa_cifar10.research.mlp_diagnostics as diagnostics
from mjepa_cifar10.pretrain import CIFAR10MJEPA
from mjepa_cifar10.research.mlp_diagnostics import (
    CompletedCheckpoint,
    InstrumentedNormMLP,
    SignalMoments,
    analyze_completed_checkpoint,
    instrument_student_mlps,
    load_checkpoint_model,
    validate_completed_checkpoint,
)


HIDDEN_SIZE = 2
FFN_HIDDEN_SIZE = 2
TINY_BACKBONE_HIDDEN_SIZE = 4
TINY_BACKBONE_FFN_HIDDEN_SIZE = 8
NEGATIVE_GATE_FRACTION = 0.5


def _identity_mlp(activation: str) -> NormMLP:
    mlp = NormMLP(
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        bias=False,
        activation=activation,
        dropout=0.0,
        norm_type="layernorm",
    )
    with torch.no_grad():
        mlp.fc2.weight.copy_(torch.eye(HIDDEN_SIZE))
        if activation.endswith("glu"):
            mlp.fc1.weight.copy_(torch.cat((torch.eye(HIDDEN_SIZE), torch.eye(HIDDEN_SIZE))))
        else:
            mlp.fc1.weight.copy_(torch.eye(HIDDEN_SIZE))
    return mlp


def _diagnose_one_mlp(activation: str) -> dict[str, float | int | str]:
    instrumented = InstrumentedNormMLP(_identity_mlp(activation), activation)
    input_tensor = torch.tensor([[[-1.0, 1.0]]], requires_grad=True)

    instrumented(input_tensor).sum().backward()

    return instrumented.summary()


def test_srelu_diagnostic_measures_clipped_negative_gradient_energy() -> None:
    summary = _diagnose_one_mlp("srelu")

    assert summary["gate_negative_fraction"] == pytest.approx(NEGATIVE_GATE_FRACTION)
    assert summary["gate_activation_zero_fraction"] == pytest.approx(NEGATIVE_GATE_FRACTION)
    assert summary["negative_gate_upstream_gradient_energy_fraction"] == pytest.approx(NEGATIVE_GATE_FRACTION)
    assert summary["negative_gate_preactivation_gradient_nonzero_fraction"] == 0.0
    assert summary["dead_channel_fraction"] == pytest.approx(NEGATIVE_GATE_FRACTION)


def test_swiglu_diagnostic_detects_gradient_survival_at_negative_gates() -> None:
    summary = _diagnose_one_mlp("swiglu")

    assert summary["gate_negative_fraction"] == pytest.approx(NEGATIVE_GATE_FRACTION)
    assert summary["gate_activation_zero_fraction"] == 0.0
    assert summary["negative_gate_upstream_gradient_energy_fraction"] == pytest.approx(NEGATIVE_GATE_FRACTION)
    assert summary["negative_gate_preactivation_gradient_nonzero_fraction"] == 1.0


@pytest.mark.parametrize("activation", ["srelu", "swiglu"])
def test_instrumented_mlp_preserves_eval_output(activation: str) -> None:
    mlp = _identity_mlp(activation).eval()
    instrumented = InstrumentedNormMLP(mlp, activation).eval()
    input_tensor = torch.tensor([[[-1.0, 1.0]]])

    with torch.no_grad():
        expected = mlp(input_tensor)
        actual = instrumented(input_tensor)

    assert torch.allclose(actual, expected)


def test_validate_completed_checkpoint_rejects_nonterminal_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()

    with pytest.raises(ValueError, match="completed terminal state"):
        validate_completed_checkpoint(checkpoint)


def test_validate_completed_checkpoint_accepts_completed_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    (tmp_path / "config.yaml").touch()
    (tmp_path / "terminal.json").write_text(
        json.dumps({"status": "completed", "exit_code": 0, "wandb_run_id": "run-id"}),
        encoding="utf-8",
    )

    completed = validate_completed_checkpoint(checkpoint)

    assert completed.checkpoint == checkpoint.resolve()
    assert completed.terminal["wandb_run_id"] == "run-id"


def test_validate_completed_checkpoint_rejects_unexpected_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "live-copy.pt"
    checkpoint.touch()
    (tmp_path / "terminal.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint.pt"):
        validate_completed_checkpoint(checkpoint)


def test_instrumented_mlp_is_a_module() -> None:
    assert isinstance(InstrumentedNormMLP(_identity_mlp("srelu"), "srelu"), nn.Module)


def test_signal_moments_are_zero_before_observations() -> None:
    moments = SignalMoments()

    assert moments.mean == 0.0
    assert moments.rms == 0.0
    assert moments.zero_fraction == 0.0


def test_instrumented_mlp_rejects_activation_projection_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        InstrumentedNormMLP(_identity_mlp("srelu"), "swiglu")


def test_instrumented_mlp_reports_bias_and_applies_modulation() -> None:
    mlp = NormMLP(
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        bias=True,
        activation="srelu",
        dropout=0.0,
        norm_type="layernorm",
    )
    instrumented = InstrumentedNormMLP(mlp, "srelu").eval()
    input_tensor = torch.tensor([[[-1.0, 1.0]]])
    scale_delta = torch.zeros_like(input_tensor)
    shift = torch.zeros_like(input_tensor)
    output_gate = torch.full_like(input_tensor, 0.5)

    output = instrumented(
        input_tensor,
        norm_scale_delta=scale_delta,
        norm_shift=shift,
        output_gate=output_gate,
    )
    summary = instrumented.summary()
    instrumented(input_tensor)

    assert output.shape == input_tensor.shape
    assert "fc1_bias_mean" in summary
    assert summary["fc1_gradient_norm"] == 0.0


def test_validate_completed_checkpoint_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_completed_checkpoint(tmp_path / "checkpoint.pt")


@pytest.mark.parametrize(
    "terminal",
    [
        {"status": "running"},
        {"status": "completed", "exit_code": 1},
        [],
    ],
)
def test_validate_completed_checkpoint_rejects_noncompleted_state(tmp_path: Path, terminal: object) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    (tmp_path / "terminal.json").write_text(json.dumps(terminal), encoding="utf-8")

    with pytest.raises(ValueError, match="completed terminal state"):
        validate_completed_checkpoint(checkpoint)


def test_validate_completed_checkpoint_requires_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    (tmp_path / "terminal.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="config.yaml"):
        validate_completed_checkpoint(checkpoint)


def _tiny_backbone_config(activation: str = "srelu") -> ViTConfig:
    return ViTConfig(
        in_channels=3,
        patch_size=[2, 2],
        img_size=[4, 4],
        depth=1,
        hidden_size=TINY_BACKBONE_HIDDEN_SIZE,
        ffn_hidden_size=TINY_BACKBONE_FFN_HIDDEN_SIZE,
        num_attention_heads=1,
        activation=activation,
        norm_type="layernorm",
        dtype=torch.float32,
    )


def test_instrument_student_mlps_wraps_every_backbone_layer() -> None:
    backbone = _tiny_backbone_config().instantiate(device=torch.device("cpu"))

    instruments = instrument_student_mlps(backbone, "srelu")

    assert len(instruments) == 1
    assert isinstance(backbone.blocks[0].mlp, InstrumentedNormMLP)


def test_load_checkpoint_model_restores_all_jepa_weights(tmp_path: Path) -> None:
    backbone_config = _tiny_backbone_config()
    jepa_config = JEPAConfig(predictor_depth=1)
    backbone = backbone_config.instantiate(device=torch.device("cpu"))
    predictor = CrossAttentionPredictor(
        backbone,
        jepa_config.predictor_depth,
        device=torch.device("cpu"),
        attention_mode=jepa_config.predictor_attention_mode,
        disable_predictor_regularizers=jepa_config.disable_predictor_regularizers,
    )
    source = CIFAR10MJEPA(jepa_config, backbone, predictor, autocast_dtype=torch.float32)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "backbone": source.student.state_dict(),
            "predictor": source.predictor.state_dict(),
            "teacher": source.teacher.state_dict(),
            "step": 7,
            "epoch": 9,
        },
        checkpoint,
    )

    restored, metadata = load_checkpoint_model(checkpoint, backbone_config, jepa_config)

    assert metadata["step"] == 7
    assert torch.equal(restored.student.cls_tokens, source.student.cls_tokens)


class _FakeJEPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.student = nn.Identity()
        self.loss_parameter = nn.Parameter(torch.tensor(2.0))

    def forward(self, images: torch.Tensor, jepa_scale: int, epoch: int) -> object:
        del images, jepa_scale, epoch
        return object()

    def compute_losses(self, output: object, step: int, epoch: int) -> SimpleNamespace:
        del output, step, epoch
        loss = self.loss_parameter.square()
        return SimpleNamespace(
            jepa_loss=loss,
            jepa_loss_cls=0.0,
            sigreg_loss=0.0,
            gram_loss=0.0,
            reduce=lambda: loss,
        )


class _FakeInstrument:
    def summary(self) -> dict[str, float | int | str]:
        return {"activation": "srelu", "sample_count": 1}


def test_analyze_completed_checkpoint_aggregates_batches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    config_path = tmp_path / "config.yaml"
    checkpoint.touch()
    config_path.write_text("placeholder", encoding="utf-8")
    completed = CompletedCheckpoint(
        checkpoint=checkpoint,
        config=config_path,
        terminal={"status": "completed", "wandb_run_id": "run-id"},
    )
    backbone_config = _tiny_backbone_config()
    jepa_config = JEPAConfig()
    fake_jepa = _FakeJEPA()
    fake_batches = [
        (torch.ones(1, 3, 4, 4), torch.zeros(1, dtype=torch.long)),
        (torch.ones(1, 3, 4, 4), torch.zeros(1, dtype=torch.long)),
    ]
    monkeypatch.setattr(diagnostics.yaml, "full_load", lambda _: {"backbone": backbone_config, "jepa": jepa_config})
    monkeypatch.setattr(diagnostics, "load_checkpoint_model", lambda *_: (fake_jepa, {"step": 7, "epoch": 9}))
    monkeypatch.setattr(diagnostics, "instrument_student_mlps", lambda *_: [cast(Any, _FakeInstrument())])
    monkeypatch.setattr(diagnostics, "get_val_dataloader", lambda **_: fake_batches)

    result = analyze_completed_checkpoint(
        completed,
        data_root=tmp_path,
        batch_size=1,
        num_batches=1,
        seed=3,
        model_mode="eval",
    )

    assert result["checkpoint_step"] == 7
    assert result["checkpoint_epoch"] == 9
    assert result["losses"]["reduced_loss"] == 4.0
    assert result["layers"] == [{"layer": 0, "activation": "srelu", "sample_count": 1}]


@pytest.mark.parametrize(
    ("batch_size", "num_batches", "model_mode", "message"),
    [
        (0, 1, "eval", "positive"),
        (1, 0, "eval", "positive"),
        (1, 1, "invalid", "model_mode"),
    ],
)
def test_analyze_completed_checkpoint_validates_arguments(
    tmp_path: Path,
    batch_size: int,
    num_batches: int,
    model_mode: str,
    message: str,
) -> None:
    completed = CompletedCheckpoint(tmp_path / "checkpoint.pt", tmp_path / "config.yaml", {})

    with pytest.raises(ValueError, match=message):
        analyze_completed_checkpoint(
            completed,
            data_root=tmp_path,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=0,
            model_mode=model_mode,
        )
