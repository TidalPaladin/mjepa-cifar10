import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from vit import ViT

from .experiment import save_safetensors_atomic


FINAL_CLS_MODE: Final[Literal["final_cls"]] = "final_cls"
LAST_TWO_CLS_MODE: Final[Literal["last_two_cls"]] = "last_two_cls"
ProbeFeatureMode = Literal["final_cls", "last_two_cls"]


def _pool_cls_tokens(dense_features: Tensor, num_cls_tokens: int) -> Tensor:
    if num_cls_tokens <= 0:
        raise ValueError("CLS probe calibration requires at least one CLS token")
    return dense_features[:, :num_cls_tokens].mean(dim=1)


def extract_probe_features(backbone: ViT, images: Tensor) -> Mapping[ProbeFeatureMode, Tensor]:
    """Extract normalized final and penultimate CLS features from a frozen ViT."""
    if len(backbone.blocks) < 2:
        raise ValueError("last-two CLS extraction requires at least two encoder layers")

    penultimate_output: Tensor | None = None

    def capture_penultimate(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal penultimate_output
        penultimate_output = output

    handle = backbone.blocks[-2].register_forward_hook(capture_penultimate)
    try:
        final_output = backbone(images).dense_features
    finally:
        handle.remove()

    if penultimate_output is None:
        raise RuntimeError("penultimate encoder layer did not produce an output")

    num_cls_tokens = backbone.config.num_cls_tokens
    penultimate_cls = _pool_cls_tokens(backbone.output_norm(penultimate_output), num_cls_tokens)
    final_cls = _pool_cls_tokens(final_output, num_cls_tokens)
    return {
        FINAL_CLS_MODE: final_cls,
        LAST_TWO_CLS_MODE: torch.cat((penultimate_cls, final_cls), dim=-1),
    }


class FrozenBackboneProbe(nn.Module):
    """Train a fresh classifier while keeping its pretrained encoder frozen and deterministic."""

    def __init__(
        self,
        backbone: ViT,
        *,
        mode: ProbeFeatureMode,
        num_classes: int,
        normalize: bool,
    ) -> None:
        super().__init__()
        if mode == LAST_TWO_CLS_MODE and len(backbone.blocks) < 2:
            raise ValueError("last-two CLS extraction requires at least two encoder layers")
        if backbone.config.num_cls_tokens <= 0:
            raise ValueError("CLS probe calibration requires at least one CLS token")

        self.backbone = backbone.requires_grad_(False)
        self.mode = mode
        feature_size = backbone.config.hidden_size * (2 if mode == LAST_TWO_CLS_MODE else 1)
        self.normalization = nn.LayerNorm(feature_size, elementwise_affine=False) if normalize else nn.Identity()
        self.classifier = nn.Linear(feature_size, num_classes)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenBackboneProbe":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        self.backbone.eval()
        with torch.no_grad():
            if self.mode == FINAL_CLS_MODE:
                dense_features = self.backbone(images).dense_features
                features = _pool_cls_tokens(dense_features, self.backbone.config.num_cls_tokens)
            else:
                features = extract_probe_features(self.backbone, images)[LAST_TWO_CLS_MODE]
        return self.classifier(self.normalization(features))


class LinearProbeBank(nn.Module):
    """Independent linear heads initialized identically for a fixed learning-rate sweep."""

    def __init__(self, *, feature_size: int, num_classes: int, learning_rates: tuple[float, ...]) -> None:
        super().__init__()
        if not learning_rates or any(learning_rate <= 0 for learning_rate in learning_rates):
            raise ValueError("learning_rates must contain positive values")
        self.learning_rates = learning_rates
        self.classifiers = nn.ModuleList(nn.Linear(feature_size, num_classes) for _ in learning_rates)
        initial_state = self.classifiers[0].state_dict()
        for classifier in self.classifiers[1:]:
            classifier.load_state_dict(initial_state)

    def forward(self, features: Tensor) -> Tensor:
        return torch.stack([classifier(features) for classifier in self.classifiers])


@dataclass(frozen=True)
class ProbeTrainingResult:
    learning_rates: tuple[float, ...]
    peak_accuracies: tuple[float, ...]
    final_accuracies: tuple[float, ...]
    validation_curves: tuple[tuple[float, ...], ...]
    best_index: int

    @property
    def best_learning_rate(self) -> float:
        return self.learning_rates[self.best_index]

    @property
    def best_peak_accuracy(self) -> float:
        return self.peak_accuracies[self.best_index]

    @property
    def best_final_accuracy(self) -> float:
        return self.final_accuracies[self.best_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "learning_rates": list(self.learning_rates),
            "peak_accuracies": list(self.peak_accuracies),
            "final_accuracies": list(self.final_accuracies),
            "validation_curves": [list(curve) for curve in self.validation_curves],
            "best_index": self.best_index,
            "best_learning_rate": self.best_learning_rate,
            "best_peak_accuracy": self.best_peak_accuracy,
            "best_final_accuracy": self.best_final_accuracy,
        }


def save_feature_cache(path: Path, tensors: Mapping[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors_atomic(path, {key: value.detach().cpu().contiguous() for key, value in tensors.items()})


def load_feature_cache(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return st.load_file(str(path), device="cpu")


def extract_dataset_features(backbone: ViT, dataloader: DataLoader, device: torch.device) -> dict[str, Tensor]:
    """Extract both preregistered CLS representations for every example in loader order."""
    backbone.requires_grad_(False)
    backbone.eval()
    final_features: list[Tensor] = []
    last_two_features: list[Tensor] = []
    labels: list[Tensor] = []
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )

    with torch.no_grad(), autocast_context:
        for images, batch_labels in dataloader:
            extracted = extract_probe_features(backbone, images.to(device, non_blocking=True))
            final_features.append(extracted[FINAL_CLS_MODE].float().cpu())
            last_two_features.append(extracted[LAST_TWO_CLS_MODE].float().cpu())
            labels.append(batch_labels.cpu())

    return {
        FINAL_CLS_MODE: torch.cat(final_features),
        LAST_TWO_CLS_MODE: torch.cat(last_two_features),
        "labels": torch.cat(labels),
    }


def _make_probe_optimizer(
    bank: LinearProbeBank,
    *,
    weight_decay: float,
) -> AdamW:
    parameter_groups = [
        {"params": classifier.parameters(), "lr": learning_rate}
        for classifier, learning_rate in zip(bank.classifiers, bank.learning_rates, strict=True)
    ]
    return AdamW(parameter_groups, weight_decay=weight_decay, betas=(0.9, 0.999))


def _probe_accuracy(bank: LinearProbeBank, features: Tensor, labels: Tensor, batch_size: int) -> tuple[float, ...]:
    correct = torch.zeros(len(bank.classifiers), dtype=torch.long, device=features.device)
    total = 0
    bank.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            end = start + batch_size
            predictions = bank(features[start:end]).argmax(dim=-1)
            batch_labels = labels[start:end]
            correct += (predictions == batch_labels.unsqueeze(0)).sum(dim=1)
            total += len(batch_labels)
    return tuple(float(value) / total for value in correct.tolist())


def train_probe_bank(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    *,
    learning_rates: tuple[float, ...],
    epochs: int,
    batch_size: int,
    weight_decay: float,
    warmup_fraction: float,
    start_factor: float,
    final_factor: float,
    device: torch.device,
    seed: int,
) -> ProbeTrainingResult:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("probe features must have shape [examples, features]")
    if len(train_features) != len(train_labels) or len(validation_features) != len(validation_labels):
        raise ValueError("probe feature and label counts must match")

    torch.manual_seed(seed)
    bank = LinearProbeBank(
        feature_size=train_features.shape[1],
        num_classes=int(torch.max(torch.cat((train_labels, validation_labels))).item()) + 1,
        learning_rates=learning_rates,
    ).to(device)
    optimizer = _make_probe_optimizer(bank, weight_decay=weight_decay)
    batches_per_epoch = math.ceil(len(train_features) / batch_size)
    total_steps = epochs * batches_per_epoch
    warmup_steps = int(total_steps * warmup_fraction)
    factor = lambda step: warmup_cosine_factor(  # noqa: E731
        step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        start_factor=start_factor,
        final_factor=final_factor,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=[factor] * len(learning_rates))

    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_features = validation_features.to(device)
    validation_labels = validation_labels.to(device)
    validation_curves = [[] for _ in learning_rates]

    for _epoch in range(epochs):
        bank.train()
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = bank(features)
            loss = torch.stack([F.cross_entropy(head_logits, labels) for head_logits in logits]).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        accuracies = _probe_accuracy(bank, validation_features, validation_labels, batch_size)
        for curve, accuracy in zip(validation_curves, accuracies, strict=True):
            curve.append(accuracy)

    curves = tuple(tuple(curve) for curve in validation_curves)
    peaks = tuple(max(curve) for curve in curves)
    finals = tuple(curve[-1] for curve in curves)
    best_index = max(range(len(learning_rates)), key=lambda index: (peaks[index], -learning_rates[index]))
    return ProbeTrainingResult(
        learning_rates=learning_rates,
        peak_accuracies=peaks,
        final_accuracies=finals,
        validation_curves=curves,
        best_index=best_index,
    )


def warmup_cosine_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    start_factor: float,
    final_factor: float,
) -> float:
    """Return a linear-warmup, cosine-decay multiplier at one optimizer step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0 < start_factor <= 1:
        raise ValueError("start_factor must be in (0, 1]")
    if not 0 <= final_factor <= 1:
        raise ValueError("final_factor must be in [0, 1]")

    bounded_step = min(max(step, 0), total_steps)
    if warmup_steps > 0 and bounded_step <= warmup_steps:
        return start_factor + (1.0 - start_factor) * bounded_step / warmup_steps

    decay_steps = total_steps - warmup_steps
    progress = (bounded_step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cast(float, final_factor + (1.0 - final_factor) * cosine)
