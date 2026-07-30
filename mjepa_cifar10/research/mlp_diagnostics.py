from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Mapping, cast

import torch
import torch.nn.functional as F
import yaml
from mjepa import JEPAConfig
from mjepa.jepa import CrossAttentionPredictor
from torch import Tensor, nn
from vit import ViT, ViTConfig
from vit.fused import NormMLP
from vit.norm import apply_norm, get_norm_bias
from vit.transformer import TransformerEncoderLayer

from mjepa_cifar10.data import get_val_dataloader
from mjepa_cifar10.pretrain import CIFAR10MJEPA


CHECKPOINT_FILENAME: Final = "checkpoint.pt"
CONFIG_FILENAME: Final = "config.yaml"
TERMINAL_FILENAME: Final = "terminal.json"
NEAR_ZERO_THRESHOLD: Final = 1e-6
GRADIENT_NONZERO_THRESHOLD: Final = 0.0


@dataclass(frozen=True)
class CompletedCheckpoint:
    checkpoint: Path
    config: Path
    terminal: Mapping[str, Any]

    @property
    def run_id(self) -> str:
        return self.checkpoint.parent.name


@dataclass
class SignalMoments:
    count: int = 0
    zero_count: int = 0
    sum_values: float = 0.0
    sum_squares: float = 0.0
    abs_max: float = 0.0

    def update(self, value: Tensor) -> None:
        detached = value.detach().float()
        self.count += detached.numel()
        self.zero_count += int(torch.count_nonzero(detached == 0).item())
        self.sum_values += float(detached.sum().item())
        self.sum_squares += float(detached.square().sum().item())
        self.abs_max = max(self.abs_max, float(detached.abs().max().item()))

    @property
    def mean(self) -> float:
        return self.sum_values / self.count if self.count else 0.0

    @property
    def rms(self) -> float:
        return math.sqrt(self.sum_squares / self.count) if self.count else 0.0

    @property
    def zero_fraction(self) -> float:
        return self.zero_count / self.count if self.count else 0.0


class MLPSignalAccumulator:
    def __init__(self) -> None:
        self.input = SignalMoments()
        self.gate = SignalMoments()
        self.gate_activation = SignalMoments()
        self.hidden = SignalMoments()
        self.output = SignalMoments()
        self.activation_upstream_gradient = SignalMoments()
        self.preactivation_gradient = SignalMoments()
        self.gate_negative_count = 0
        self.gate_nonpositive_count = 0
        self.gate_near_zero_count = 0
        self.negative_upstream_gradient_energy = 0.0
        self.negative_preactivation_gradient_energy = 0.0
        self.negative_preactivation_gradient_nonzero_count = 0
        self.positive_channel_seen: Tensor | None = None

    def record_forward(
        self,
        input_tensor: Tensor,
        gate: Tensor,
        gate_activation: Tensor,
        hidden: Tensor,
        output: Tensor,
    ) -> None:
        self.input.update(input_tensor)
        self.gate.update(gate)
        self.gate_activation.update(gate_activation)
        self.hidden.update(hidden)
        self.output.update(output)
        detached_gate = gate.detach()
        self.gate_negative_count += int(torch.count_nonzero(detached_gate < 0).item())
        self.gate_nonpositive_count += int(torch.count_nonzero(detached_gate <= 0).item())
        self.gate_near_zero_count += int(torch.count_nonzero(detached_gate.abs() <= NEAR_ZERO_THRESHOLD).item())
        positive_seen = (detached_gate > 0).flatten(end_dim=-2).any(dim=0).cpu()
        if self.positive_channel_seen is None:
            self.positive_channel_seen = positive_seen
        else:
            self.positive_channel_seen.logical_or_(positive_seen)

    def record_activation_gradient(self, gradient: Tensor, negative_gate: Tensor) -> None:
        self.activation_upstream_gradient.update(gradient)
        detached_gradient = gradient.detach().float()
        self.negative_upstream_gradient_energy += float(detached_gradient[negative_gate].square().sum().item())

    def record_preactivation_gradient(self, gradient: Tensor, negative_gate: Tensor) -> None:
        self.preactivation_gradient.update(gradient)
        detached_gradient = gradient.detach().float()
        negative_gradient = detached_gradient[negative_gate]
        self.negative_preactivation_gradient_energy += float(negative_gradient.square().sum().item())
        self.negative_preactivation_gradient_nonzero_count += int(
            torch.count_nonzero(negative_gradient.abs() > GRADIENT_NONZERO_THRESHOLD).item()
        )

    def summary(self) -> dict[str, float | int]:
        gate_count = self.gate.count
        negative_count = self.gate_negative_count
        upstream_energy = self.activation_upstream_gradient.sum_squares
        preactivation_energy = self.preactivation_gradient.sum_squares
        dead_channel_fraction = (
            float((~self.positive_channel_seen).float().mean().item())
            if self.positive_channel_seen is not None
            else 0.0
        )
        return {
            "sample_count": gate_count,
            "gate_negative_fraction": _safe_ratio(negative_count, gate_count),
            "gate_nonpositive_fraction": _safe_ratio(self.gate_nonpositive_count, gate_count),
            "gate_near_zero_fraction": _safe_ratio(self.gate_near_zero_count, gate_count),
            "gate_mean": self.gate.mean,
            "gate_rms": self.gate.rms,
            "gate_abs_max": self.gate.abs_max,
            "gate_activation_mean": self.gate_activation.mean,
            "gate_activation_zero_fraction": self.gate_activation.zero_fraction,
            "gate_activation_rms": self.gate_activation.rms,
            "dead_channel_fraction": dead_channel_fraction,
            "input_rms": self.input.rms,
            "hidden_rms": self.hidden.rms,
            "output_rms": self.output.rms,
            "output_to_input_rms_ratio": _safe_ratio(self.output.rms, self.input.rms),
            "activation_upstream_gradient_rms": self.activation_upstream_gradient.rms,
            "negative_gate_upstream_gradient_energy_fraction": _safe_ratio(
                self.negative_upstream_gradient_energy, upstream_energy
            ),
            "preactivation_gradient_rms": self.preactivation_gradient.rms,
            "negative_gate_preactivation_gradient_energy_fraction": _safe_ratio(
                self.negative_preactivation_gradient_energy, preactivation_energy
            ),
            "negative_gate_preactivation_gradient_nonzero_fraction": _safe_ratio(
                self.negative_preactivation_gradient_nonzero_count, negative_count
            ),
        }


class InstrumentedNormMLP(nn.Module):
    """Eager equivalent of ``NormMLP`` that records gate signals and gradients."""

    def __init__(self, mlp: NormMLP, activation_name: str):
        super().__init__()
        if activation_name.endswith("glu") != (mlp.fc1.out_features == 2 * mlp.fc2.in_features):
            raise ValueError(f"activation {activation_name!r} does not match the MLP projection shape")
        self.mlp = mlp
        self.activation_name = activation_name
        self.is_glu = activation_name.endswith("glu")
        self.signals = MLPSignalAccumulator()

    def forward(
        self,
        x: Tensor,
        *,
        norm_scale_delta: Tensor | None = None,
        norm_shift: Tensor | None = None,
        output_gate: Tensor | None = None,
    ) -> Tensor:
        normalized = apply_norm(
            x,
            self.mlp.norm.weight,
            get_norm_bias(self.mlp.norm),
            self.mlp.norm.eps or 1e-5,
            use_layer_norm=isinstance(self.mlp.norm, nn.LayerNorm),
            scale_delta=norm_scale_delta,
            shift=norm_shift,
        )
        projected = F.linear(normalized, self.mlp.fc1.weight, self.mlp.fc1.bias)
        if self.is_glu:
            linear, gate = projected.chunk(2, dim=-1)
            if self.mlp.limit is not None:
                linear = linear.clamp(min=-self.mlp.limit, max=self.mlp.limit)
                gate = gate.clamp(max=self.mlp.limit)
            if self.mlp.extra_bias is not None:
                linear = linear + self.mlp.extra_bias
            gate_activation = _activate(gate, self.activation_name)
            hidden = gate_activation * linear
        else:
            gate = projected
            gate_activation = _activate(gate, self.activation_name)
            hidden = gate_activation

        negative_gate = gate.detach() < 0
        if gate_activation.requires_grad:
            gate_activation.register_hook(
                lambda gradient: self.signals.record_activation_gradient(gradient, negative_gate)
            )
        if gate.requires_grad:
            gate.register_hook(lambda gradient: self.signals.record_preactivation_gradient(gradient, negative_gate))

        hidden_after_dropout = F.dropout(hidden, p=self.mlp.dropout.p, training=self.training)
        output = F.linear(hidden_after_dropout, self.mlp.fc2.weight, self.mlp.fc2.bias)
        output = F.dropout(output, p=self.mlp.dropout.p, training=self.training)
        if output_gate is not None:
            output = output * output_gate
        self.signals.record_forward(x, gate, gate_activation, hidden, output)
        return output

    def summary(self) -> dict[str, float | int | str]:
        summary: dict[str, float | int | str] = {
            "activation": self.activation_name,
            "ffn_hidden_size": self.mlp.fc2.in_features,
            **self.signals.summary(),
            "fc1_weight_norm": float(self.mlp.fc1.weight.detach().float().norm().item()),
            "fc2_weight_norm": float(self.mlp.fc2.weight.detach().float().norm().item()),
            "fc1_gradient_norm": _gradient_norm(self.mlp.fc1.weight),
            "fc2_gradient_norm": _gradient_norm(self.mlp.fc2.weight),
        }
        if self.mlp.fc1.bias is not None:
            summary.update(
                {
                    "fc1_bias_mean": float(self.mlp.fc1.bias.detach().float().mean().item()),
                    "fc1_bias_min": float(self.mlp.fc1.bias.detach().float().min().item()),
                    "fc1_bias_max": float(self.mlp.fc1.bias.detach().float().max().item()),
                }
            )
        return summary


def validate_completed_checkpoint(checkpoint: Path) -> CompletedCheckpoint:
    resolved = checkpoint.resolve()
    if resolved.name != CHECKPOINT_FILENAME:
        raise ValueError(f"diagnostics require the canonical {CHECKPOINT_FILENAME} path")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    terminal_path = resolved.parent / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise ValueError(f"checkpoint lacks a completed terminal state: {terminal_path}")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not isinstance(terminal, dict) or terminal.get("status") != "completed" or terminal.get("exit_code", 0) != 0:
        raise ValueError(f"checkpoint lacks a completed terminal state: {terminal_path}")
    config_path = resolved.parent / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return CompletedCheckpoint(resolved, config_path.resolve(), terminal)


def analyze_completed_checkpoint(
    completed: CompletedCheckpoint,
    *,
    data_root: Path,
    batch_size: int,
    num_batches: int,
    seed: int,
    model_mode: str,
) -> dict[str, Any]:
    if batch_size < 1 or num_batches < 1:
        raise ValueError("batch_size and num_batches must be positive")
    if model_mode not in {"eval", "train"}:
        raise ValueError("model_mode must be 'eval' or 'train'")
    config = yaml.full_load(completed.config.read_text(encoding="utf-8"))
    backbone_config = config.get("backbone")
    jepa_config = config.get("jepa")
    if not isinstance(backbone_config, ViTConfig) or not isinstance(jepa_config, JEPAConfig):
        raise TypeError("run config must contain ViTConfig 'backbone' and JEPAConfig 'jepa' objects")

    started_at = perf_counter()
    jepa, checkpoint_metadata = load_checkpoint_model(completed.checkpoint, backbone_config, jepa_config)
    instruments = instrument_student_mlps(jepa.student, backbone_config.activation)
    jepa.train(model_mode == "train")
    dataloader = get_val_dataloader(
        size=backbone_config.img_size,
        batch_size=batch_size,
        root=data_root,
        num_workers=0,
    )
    loss_sums: dict[str, float] = {}
    processed_batches = 0
    with torch.compiler.set_stance("force_eager"):
        for batch_index, (images, _) in enumerate(dataloader):
            if batch_index >= num_batches:
                break
            jepa.zero_grad(set_to_none=True)
            torch.manual_seed(seed + batch_index)
            output = jepa(images, jepa_config.scale, int(checkpoint_metadata["epoch"]))
            losses = jepa.compute_losses(
                output,
                int(checkpoint_metadata["step"]),
                int(checkpoint_metadata["epoch"]),
            )
            reduced_loss = losses.reduce()
            reduced_loss.backward()
            for name, value in (
                ("jepa_loss", losses.jepa_loss),
                ("jepa_loss_cls", losses.jepa_loss_cls),
                ("sigreg_loss", losses.sigreg_loss),
                ("gram_loss", losses.gram_loss),
                ("reduced_loss", reduced_loss),
            ):
                loss_sums[name] = loss_sums.get(name, 0.0) + _scalar_value(value)
            processed_batches += 1
    if processed_batches != num_batches:
        raise RuntimeError(f"requested {num_batches} batches but only processed {processed_batches}")
    return {
        "schema_version": 1,
        "run_id": completed.run_id,
        "checkpoint": str(completed.checkpoint),
        "config": str(completed.config),
        "wandb_run_id": completed.terminal.get("wandb_run_id"),
        "checkpoint_step": int(checkpoint_metadata["step"]),
        "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
        "activation": backbone_config.activation,
        "ffn_hidden_size": backbone_config.ffn_hidden_size,
        "model_mode": model_mode,
        "device": "cpu",
        "autocast_dtype": "float32",
        "dataset_split": "fixed-validation-holdout",
        "batch_size": batch_size,
        "num_batches": num_batches,
        "seed": seed,
        "losses": {name: total / processed_batches for name, total in loss_sums.items()},
        "layers": [
            {
                "layer": layer_index,
                **instrument.summary(),
            }
            for layer_index, instrument in enumerate(instruments)
        ],
        "elapsed_seconds": perf_counter() - started_at,
    }


def instrument_student_mlps(student: ViT, activation_name: str) -> list[InstrumentedNormMLP]:
    instruments: list[InstrumentedNormMLP] = []
    for layer_index, block in enumerate(student.blocks):
        if not isinstance(block, TransformerEncoderLayer) or not isinstance(block.mlp, NormMLP):
            raise TypeError(f"student layer {layer_index} does not contain a supported NormMLP")
        instrument = InstrumentedNormMLP(block.mlp, activation_name)
        block.mlp = cast(Any, instrument)
        instruments.append(instrument)
    return instruments


def load_checkpoint_model(
    checkpoint: Path,
    backbone_config: ViTConfig,
    jepa_config: JEPAConfig,
) -> tuple[CIFAR10MJEPA, Mapping[str, Any]]:
    device = torch.device("cpu")
    backbone = backbone_config.instantiate(device=device)
    predictor = CrossAttentionPredictor(
        backbone,
        jepa_config.predictor_depth,
        device=device,
        attention_mode=jepa_config.predictor_attention_mode,
        cls_prediction_mode=jepa_config.cls_prediction_mode,
        cls_context_tokens=jepa_config.cls_context_tokens,
        disable_predictor_regularizers=jepa_config.disable_predictor_regularizers,
    )
    jepa = CIFAR10MJEPA(jepa_config, backbone, predictor, autocast_dtype=torch.float32)
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(data, dict):
        raise TypeError("checkpoint must contain a mapping")
    jepa.student.load_state_dict(data["backbone"])
    jepa.predictor.load_state_dict(data["predictor"])
    teacher_state = data.get("teacher")
    if jepa.teacher is not None:
        if not isinstance(teacher_state, Mapping):
            raise TypeError("EMA checkpoint must contain a teacher state mapping")
        jepa.teacher.load_state_dict(teacher_state)
    elif teacher_state is not None:
        raise ValueError("shared-target configuration cannot load an EMA teacher state")
    return jepa, data


def _activate(value: Tensor, activation_name: str) -> Tensor:
    match activation_name:
        case "srelu":
            return F.relu(value).square()
        case "swiglu" | "silu":
            return F.silu(value)
        case "geglu" | "gelu":
            return F.gelu(value)
        case "reglu" | "relu":
            return F.relu(value)
        case _:
            raise ValueError(f"unsupported diagnostic activation: {activation_name}")


def _gradient_norm(parameter: Tensor) -> float:
    return float(parameter.grad.detach().float().norm().item()) if parameter.grad is not None else 0.0


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _scalar_value(value: Tensor | float) -> float:
    return float(value.detach().item()) if isinstance(value, Tensor) else float(value)
