#!/usr/bin/env python3

import hashlib
import json
import os
import socket
import subprocess
import tempfile
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import torch
import yaml
from mjepa.metrics import CLSPatchAlignmentMetric
from torch import Tensor
from vit import ViTConfig

import wandb
from mjepa_cifar10.collapse import EmbeddingCollapseMetric, PatchTokenDiversityMetric
from mjepa_cifar10.data import cifar10_split_fingerprint, get_probe_train_dataloader, get_val_dataloader
from mjepa_cifar10.finetune import load_backbone_checkpoint
from mjepa_cifar10.representation_diagnostics import (
    CLS_PATCH_MEAN_ROUTE,
    CLS_ROUTE,
    PATCH_MEAN_ROUTE,
    REPRESENTATION_ROUTES,
    NearestCentroidProbe,
    extract_normalized_layer_features,
    representation_routes,
)


NUM_CLASSES: Final[int] = 10
EXPECTED_EMITTED_CLASSES: Final[frozenset[str]] = frozenset(("configs", "metrics", "provenance"))
HASH_CHUNK_SIZE: Final[int] = 1024 * 1024


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Diagnose layerwise global and spatial representations in retained ViTs")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
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


def _git_sha(repository: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("representation diagnostic manifest must contain a mapping")
    for key in ("id", "sources", "data", "diagnostics", "resources", "wandb", "log_root"):
        if key not in manifest:
            raise ValueError(f"representation diagnostic manifest is missing {key!r}")
    wandb_config = manifest["wandb"]
    emitted_classes = frozenset(wandb_config.get("emitted_data_classes", {}).get("launch", ()))
    approved_classes = frozenset(wandb_config.get("approved_data_classes", ()))
    if not wandb_config.get("authorized"):
        raise ValueError("online W&B representation diagnostics are not authorized")
    if emitted_classes != EXPECTED_EMITTED_CLASSES or not emitted_classes <= approved_classes:
        raise ValueError("W&B launch manifest must authorize configs, metrics, and provenance")
    if manifest["data"].get("official_test_set") != "prohibited":
        raise ValueError("representation diagnostics must prohibit the official test set")
    return manifest, _sha256(path)


def _probe_feature_dims(hidden_size: int) -> dict[str, int]:
    return {
        CLS_ROUTE: hidden_size,
        PATCH_MEAN_ROUTE: hidden_size,
        CLS_PATCH_MEAN_ROUTE: hidden_size * 2,
    }


def _new_layer_probes(
    depth: int,
    hidden_size: int,
    device: torch.device,
) -> dict[tuple[int, str], NearestCentroidProbe]:
    dimensions = _probe_feature_dims(hidden_size)
    return {
        (layer_index, route): NearestCentroidProbe(
            num_classes=NUM_CLASSES,
            feature_dim=dimensions[route],
            device=device,
        )
        for layer_index in range(depth)
        for route in REPRESENTATION_ROUTES
    }


def _new_layer_metrics(
    depth: int,
    hidden_size: int,
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    return {
        layer_index: {
            "cpa": CLSPatchAlignmentMetric().to(device),
            "cls_collapse": EmbeddingCollapseMetric(hidden_size).to(device),
            "patch_mean_collapse": EmbeddingCollapseMetric(hidden_size).to(device),
            "patch_diversity": PatchTokenDiversityMetric(hidden_size).to(device),
        }
        for layer_index in range(depth)
    }


def _extract_train_centroids(
    backbone: Any,
    dataloader: Any,
    probes: dict[tuple[int, str], NearestCentroidProbe],
    device: torch.device,
) -> None:
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for images, labels in dataloader:
            layers = extract_normalized_layer_features(backbone, images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            for layer_index, features in enumerate(layers):
                for route, route_features in representation_routes(features).items():
                    probes[(layer_index, route)].update_train(route_features, labels)
    for probe in probes.values():
        probe.finalize()


def _evaluate_layers(
    backbone: Any,
    dataloader: Any,
    probes: dict[tuple[int, str], NearestCentroidProbe],
    metrics: dict[int, dict[str, Any]],
    device: torch.device,
) -> None:
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for images, labels in dataloader:
            layers = extract_normalized_layer_features(backbone, images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            for layer_index, features in enumerate(layers):
                routes = representation_routes(features)
                for route, route_features in routes.items():
                    probes[(layer_index, route)].update_validation(route_features, labels)
                layer_metrics = metrics[layer_index]
                layer_metrics["cpa"].update(features.cls_tokens, features.visual_tokens)
                layer_metrics["cls_collapse"].update(routes[CLS_ROUTE])
                layer_metrics["patch_mean_collapse"].update(routes[PATCH_MEAN_ROUTE])
                layer_metrics["patch_diversity"].update(features.visual_tokens)


def _tensor_metrics(values: Mapping[str, Tensor]) -> dict[str, float]:
    return {key: float(value.item()) for key, value in values.items()}


def _collect_layer_results(
    probes: dict[tuple[int, str], NearestCentroidProbe],
    metrics: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for layer_index, layer_metrics in metrics.items():
        results.append(
            {
                "layer": layer_index + 1,
                "centroid_accuracy": {
                    route: probes[(layer_index, route)].compute_accuracy() for route in REPRESENTATION_ROUTES
                },
                "cls_patch_alignment": _tensor_metrics(layer_metrics["cpa"].compute()),
                "cls_collapse": _tensor_metrics(layer_metrics["cls_collapse"].compute()),
                "patch_mean_collapse": _tensor_metrics(layer_metrics["patch_mean_collapse"].compute()),
                "patch_diversity": _tensor_metrics(layer_metrics["patch_diversity"].compute()),
            }
        )
    return results


def _diagnose_source(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_hash: str,
    source: dict[str, Any],
    data_path: Path,
    device: torch.device,
    physical_gpu: int,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    study_id = str(manifest["id"])
    source_id = str(source["id"])
    run_dir = repository / str(manifest["log_root"]) / study_id / "runs" / source_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.is_file():
        existing_result = json.loads(result_path.read_text())
        if existing_result.get("status") == "completed" and existing_result.get("manifest_sha256") == manifest_hash:
            return
        raise RuntimeError(f"existing representation diagnostic result does not match active manifest: {result_path}")

    source_run_dir = repository / str(source["run_dir"])
    checkpoint_path = source_run_dir / "backbone.safetensors"
    config_path = source_run_dir / "config.yaml"
    config = yaml.full_load(config_path.read_text())
    backbone_config = config.get("backbone") if isinstance(config, Mapping) else None
    if not isinstance(backbone_config, ViTConfig):
        raise TypeError(f"source config {config_path} does not contain a ViTConfig backbone")

    backbone = backbone_config.instantiate(device=device)
    load_backbone_checkpoint(checkpoint_path, backbone, device)
    backbone.requires_grad_(False)
    backbone.eval()

    diagnostic_config = manifest["diagnostics"]
    batch_size = int(diagnostic_config["batch_size"])
    num_workers = int(diagnostic_config["num_workers"])
    train_loader = get_probe_train_dataloader(backbone_config.img_size, batch_size, data_path, num_workers)
    validation_loader = get_val_dataloader(backbone_config.img_size, batch_size, data_path, num_workers)

    initialized_run = wandb.init(
        entity=str(manifest["wandb"]["entity"]),
        project=str(manifest["wandb"]["project"]),
        group=str(manifest["wandb"]["group"]),
        name=source_id,
        mode="online",
        dir=run_dir,
        tags=("representation-diagnostic", str(source["role"])),
        config={
            "study_id": study_id,
            "source": source,
            "diagnostics": diagnostic_config,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "emitted_data_classes": manifest["wandb"]["emitted_data_classes"]["launch"],
        },
    )
    started_at = perf_counter()
    exit_code = 0
    try:
        probes = _new_layer_probes(len(backbone.blocks), backbone_config.hidden_size, device)
        layer_metrics = _new_layer_metrics(len(backbone.blocks), backbone_config.hidden_size, device)
        _extract_train_centroids(backbone, train_loader, probes, device)
        _evaluate_layers(backbone, validation_loader, probes, layer_metrics, device)
        layers = _collect_layer_results(probes, layer_metrics)
        for layer in layers:
            layer_number = int(layer["layer"])
            wandb.log(
                {
                    **{
                        f"diagnostics/centroid_accuracy/{route}": accuracy
                        for route, accuracy in layer["centroid_accuracy"].items()
                    },
                    **{
                        f"diagnostics/cls_patch_alignment/{key}": value
                        for key, value in layer["cls_patch_alignment"].items()
                    },
                    **{f"diagnostics/patch_diversity/{key}": value for key, value in layer["patch_diversity"].items()},
                },
                step=layer_number,
            )

        result = {
            "status": "completed",
            "schema_version": 1,
            "study_id": study_id,
            "source_id": source_id,
            "source_role": source["role"],
            "manifest_sha256": manifest_hash,
            "completed_at": datetime.now(UTC).isoformat(),
            "active_seconds": perf_counter() - started_at,
            "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": _sha256(checkpoint_path)},
            "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
            "dataset": {
                "split": "fixed-45000-train-5000-validation",
                "split_sha256": cifar10_split_fingerprint(data_path),
                "official_test_set_used": False,
            },
            "evaluation": {
                "model_mode": "eval",
                "gradient_mode": "torch.inference_mode",
                "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
                "centroid_probe": "layernorm-cosine-nearest-centroid",
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "provenance": {
                "hostname": socket.gethostname(),
                "physical_gpu": physical_gpu,
                "code_shas": {
                    "parent": _git_sha(repository),
                    "mjepa": _git_sha(repository.parent / "mjepa"),
                    "vit": _git_sha(repository.parent / "vit"),
                },
                "tracker": {
                    "provider": "wandb",
                    "requested_mode": "online",
                    "effective_mode": "online",
                    "authorized": True,
                    "destination": f"{manifest['wandb']['entity']}/{manifest['wandb']['project']}",
                    "emitted_data_classes": manifest["wandb"]["emitted_data_classes"]["launch"],
                    "run_id": initialized_run.id,
                    "url": initialized_run.url,
                },
            },
            "layers": layers,
        }
        _write_json_atomic(result_path, result)
        _write_json_atomic(run_dir / "terminal.json", {"status": "completed", "result": str(result_path)})
        initialized_run.summary.update(
            {
                "diagnostics/final_centroid_cls": layers[-1]["centroid_accuracy"][CLS_ROUTE],
                "diagnostics/final_centroid_patch_mean": layers[-1]["centroid_accuracy"][PATCH_MEAN_ROUTE],
                "diagnostics/final_centroid_cls_patch_mean": layers[-1]["centroid_accuracy"][CLS_PATCH_MEAN_ROUTE],
                "diagnostics/final_cpa_mean": layers[-1]["cls_patch_alignment"]["cpa_mean"],
                "diagnostics/final_centered_patch_energy_ratio": layers[-1]["patch_diversity"][
                    "centered_patch_energy_ratio"
                ],
            }
        )
    except Exception as error:
        exit_code = 1
        _write_json_atomic(run_dir / "terminal.json", {"status": "failed", "error": str(error)[:2000]})
        raise
    finally:
        wandb.finish(exit_code=exit_code)


def main(args: Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest, manifest_hash = _load_manifest(manifest_path)
    if not args.data.is_dir():
        raise NotADirectoryError(args.data)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostics requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for source in manifest["sources"]:
        _diagnose_source(
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            source=source,
            data_path=args.data.resolve(),
            device=device,
            physical_gpu=args.physical_gpu,
        )


if __name__ == "__main__":
    main(parse_args())
