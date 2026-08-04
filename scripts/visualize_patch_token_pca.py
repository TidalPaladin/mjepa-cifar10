#!/usr/bin/env python3

import base64
import hashlib
import io
import json
import os
import tempfile
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from vit import ViTConfig

from mjepa_cifar10.data import build_stratified_split_indices, get_val_dataloader, get_val_transforms, split_fingerprint
from mjepa_cifar10.finetune import load_backbone_checkpoint
from mjepa_cifar10.patch_spatial_diagnostics import (
    PCARGBBasis,
    distance_binned_cosine_sums,
    fit_centered_pca_rgb,
    transform_centered_pca_rgb,
)
from mjepa_cifar10.representation_diagnostics import extract_normalized_layer_features


FIT_EXAMPLES_PER_CLASS: Final[int] = 64
DISPLAY_EXAMPLES_PER_CLASS: Final[int] = 1
NUM_WORKERS: Final[int] = 4
HASH_CHUNK_SIZE: Final[int] = 1024 * 1024


@dataclass(frozen=True)
class Source:
    source_id: str
    label: str
    run_dir: Path


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Create PCA maps and spatial-coherence diagnostics for retained ViTs")
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("specialized_run", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_backbone_config(run_dir: Path) -> ViTConfig:
    config_path = run_dir / "config.yaml"
    config = yaml.full_load(config_path.read_text())
    backbone_config = config.get("backbone") if isinstance(config, Mapping) else None
    if not isinstance(backbone_config, ViTConfig):
        raise TypeError(f"source config {config_path} does not contain a ViTConfig backbone")
    return backbone_config


def _stratified_diagnostic_indices(targets: Sequence[int]) -> tuple[list[int], list[int]]:
    split = build_stratified_split_indices(targets)
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index in split.validation_indices:
        indices_by_class[int(targets[index])].append(index)
    fit_indices: list[int] = []
    display_indices: list[int] = []
    for class_index in sorted(indices_by_class):
        class_indices = indices_by_class[class_index]
        required = FIT_EXAMPLES_PER_CLASS + DISPLAY_EXAMPLES_PER_CLASS
        if len(class_indices) < required:
            raise ValueError(f"class {class_index} has fewer than {required} validation examples")
        fit_indices.extend(class_indices[:FIT_EXAMPLES_PER_CLASS])
        display_indices.extend(class_indices[FIT_EXAMPLES_PER_CLASS:required])
    return sorted(fit_indices), display_indices


def _extract_visual_tokens(backbone: Any, images: Tensor, device: torch.device) -> tuple[Tensor, tuple[int, int]]:
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        features = extract_normalized_layer_features(backbone, images.to(device, non_blocking=True))[-1]
    if features.tokenized_size is None:
        raise ValueError("backbone did not report its visual-token grid size")
    tokenized_size = tuple(int(size) for size in features.tokenized_size)
    if len(tokenized_size) != 2:
        raise ValueError(f"expected a two-dimensional token grid, got {tokenized_size}")
    return features.visual_tokens.float(), (tokenized_size[0], tokenized_size[1])


def _collect_fit_tokens(
    backbone: Any,
    dataset: CIFAR10,
    fit_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, tuple[int, int]]:
    loader = DataLoader(
        Subset(dataset, list(fit_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    batches: list[Tensor] = []
    tokenized_size: tuple[int, int] | None = None
    for images, _ in loader:
        tokens, current_size = _extract_visual_tokens(backbone, images, device)
        if tokenized_size is not None and current_size != tokenized_size:
            raise ValueError("token grid changed between batches")
        tokenized_size = current_size
        batches.append(tokens)
    if tokenized_size is None:
        raise ValueError("PCA fit set is empty")
    return torch.cat(batches), tokenized_size


def _collect_spatial_curves(
    backbone: Any,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[dict[str, Any], tuple[int, int]]:
    aggregate: dict[str, tuple[Tensor | None, Tensor | None]] = {
        "raw": (None, None),
        "within_image_centered": (None, None),
    }
    tokenized_size: tuple[int, int] | None = None
    for images, _ in loader:
        tokens, current_size = _extract_visual_tokens(backbone, images, device)
        tokenized_size = current_size
        for name, center in (("raw", False), ("within_image_centered", True)):
            sums, counts = distance_binned_cosine_sums(tokens, current_size, center_within_image=center)
            previous_sums, previous_counts = aggregate[name]
            aggregate[name] = (
                sums if previous_sums is None else previous_sums + sums,
                counts if previous_counts is None else previous_counts + counts,
            )
    if tokenized_size is None:
        raise ValueError("validation loader is empty")

    curves: dict[str, Any] = {}
    for name, (sums, counts) in aggregate.items():
        if sums is None or counts is None:
            raise RuntimeError("spatial accumulator was not initialized")
        means = sums / counts
        nonadjacent_mean = sums[1:].sum() / counts[1:].sum()
        curves[name] = {
            "distance": list(range(1, means.numel() + 1)),
            "mean_cosine": means.cpu().tolist(),
            "adjacent_cosine": float(means[0].item()),
            "nonadjacent_cosine": float(nonadjacent_mean.item()),
            "neighbor_excess": float((means[0] - nonadjacent_mean).item()),
        }
    return curves, tokenized_size


def _png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _pca_images_data_uris(colors: Tensor) -> list[str]:
    return [_png_data_uri(Image.fromarray(image.cpu().numpy(), mode="RGB")) for image in colors]


def _source_result(
    source: Source,
    transformed_dataset: CIFAR10,
    validation_loader: DataLoader[Any],
    fit_indices: Sequence[int],
    display_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = source.run_dir / "backbone.safetensors"
    backbone_config = _load_backbone_config(source.run_dir)
    backbone = backbone_config.instantiate(device=device)
    load_backbone_checkpoint(checkpoint_path, backbone, device)
    backbone.requires_grad_(False)
    backbone.eval()

    fit_tokens, tokenized_size = _collect_fit_tokens(
        backbone,
        transformed_dataset,
        fit_indices,
        batch_size,
        device,
    )
    basis: PCARGBBasis = fit_centered_pca_rgb(fit_tokens)
    display_images = torch.stack([transformed_dataset[index][0] for index in display_indices])
    display_tokens, display_size = _extract_visual_tokens(backbone, display_images, device)
    if display_size != tokenized_size:
        raise ValueError("display token grid does not match PCA fit token grid")
    colors = transform_centered_pca_rgb(display_tokens, tokenized_size, basis)
    spatial_curves, spatial_size = _collect_spatial_curves(backbone, validation_loader, device)
    if spatial_size != tokenized_size:
        raise ValueError("validation token grid does not match PCA fit token grid")

    result = {
        "id": source.source_id,
        "label": source.label,
        "run_dir": str(source.run_dir),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "tokenized_size": list(tokenized_size),
        "pca_explained_variance_ratio": basis.explained_variance_ratio.cpu().tolist(),
        "pca_maps": _pca_images_data_uris(colors),
        "spatial_curves": spatial_curves,
    }
    del backbone, fit_tokens, display_tokens, basis
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    raw_dataset = CIFAR10(root=args.data, train=True, download=False)
    split = build_stratified_split_indices(raw_dataset.targets)
    fit_indices, display_indices = _stratified_diagnostic_indices(raw_dataset.targets)
    sources = (
        Source("baseline", "I-JEPA baseline", args.baseline_run.resolve()),
        Source("specialized", "Separate CLS / visual paths", args.specialized_run.resolve()),
    )
    backbone_configs = [_load_backbone_config(source.run_dir) for source in sources]
    image_sizes = {tuple(config.img_size) for config in backbone_configs}
    if len(image_sizes) != 1:
        raise ValueError(f"source image sizes differ: {sorted(image_sizes)}")
    image_size = next(iter(image_sizes))
    transformed_dataset = CIFAR10(
        root=args.data,
        train=True,
        transform=get_val_transforms(image_size),
        download=False,
    )
    validation_loader = get_val_dataloader(
        image_size,
        args.batch_size,
        args.data,
        num_workers=NUM_WORKERS,
    )

    results = [
        _source_result(
            source,
            transformed_dataset,
            validation_loader,
            fit_indices,
            display_indices,
            args.batch_size,
            device,
        )
        for source in sources
    ]
    examples = [
        {
            "dataset_index": index,
            "class_index": int(raw_dataset.targets[index]),
            "class_name": raw_dataset.classes[int(raw_dataset.targets[index])],
            "input_image": _png_data_uri(raw_dataset[index][0]),
        }
        for index in display_indices
    ]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "exploratory": True,
        "interpretation": {
            "pca_basis": "independent per model; compare spatial organization, not literal colors across models",
            "pca_centering": "subtract each image's visual-token mean before fitting and projection",
            "spatial_metric": "cosine similarity grouped by Manhattan distance on the visual-token grid",
        },
        "validation": {
            "dataset": "CIFAR-10 official training split fixed validation holdout",
            "official_test_set": "prohibited",
            "split_fingerprint": split_fingerprint(split),
            "fit_examples": len(fit_indices),
            "fit_examples_per_class": FIT_EXAMPLES_PER_CLASS,
            "display_examples": len(display_indices),
            "display_disjoint_from_fit": not bool(set(fit_indices) & set(display_indices)),
            "evaluation_examples": len(split.validation_indices),
            "model_mode": "eval",
            "gradient_mode": "torch.inference_mode",
        },
        "examples": examples,
        "sources": results,
    }
    _write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
