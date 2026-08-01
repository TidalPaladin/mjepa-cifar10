#!/usr/bin/env python3

import hashlib
import json
import os
import socket
import subprocess
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
from typing import Any, Final, cast

import torch
import torch.nn.functional as F
import yaml
from mjepa.trainer import seed_everything
from vit import ViTConfig

import wandb
from mjepa_cifar10.data import get_probe_train_dataloader, get_val_dataloader
from mjepa_cifar10.finetune import load_backbone_checkpoint
from mjepa_cifar10.probe_calibration import (
    FINAL_CLS_MODE,
    LAST_TWO_CLS_MODE,
    ProbeFeatureMode,
    ProbeTrainingResult,
    extract_dataset_features,
    load_feature_cache,
    save_feature_cache,
    train_probe_bank,
)


NUM_CLASSES: Final[int] = 10
EXPECTED_EMITTED_CLASSES: Final[frozenset[str]] = frozenset(("configs", "metrics", "provenance"))


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Calibrate frozen linear probes on retained JEPA checkpoints")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = ("id", "sources", "recipes", "probe", "wandb", "log_root")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"probe calibration manifest is missing keys: {missing}")
    wandb_config = manifest["wandb"]
    emitted_classes = frozenset(wandb_config.get("emitted_data_classes", {}).get("launch", ()))
    approved_classes = frozenset(wandb_config.get("approved_data_classes", ()))
    if not wandb_config.get("authorized"):
        raise ValueError("online W&B probe calibration is not authorized")
    if emitted_classes != EXPECTED_EMITTED_CLASSES or not emitted_classes <= approved_classes:
        raise ValueError("W&B launch manifest must authorize configs, metrics, and provenance")


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("probe calibration manifest must contain a mapping")
    _validate_manifest(manifest)
    return manifest, _sha256(path)


def _prepare_feature_cache(
    *,
    run_dir: Path,
    source_run_dir: Path,
    data_path: Path,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    checkpoint_path = source_run_dir / "backbone.safetensors"
    source_config_path = source_run_dir / "config.yaml"
    checkpoint_hash = _sha256(checkpoint_path)
    config_hash = _sha256(source_config_path)
    expected_metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "config_path": str(source_config_path.resolve()),
        "config_sha256": config_hash,
    }
    cache_path = run_dir / "features.safetensors"
    metadata_path = run_dir / "feature-cache.json"
    if cache_path.is_file() or metadata_path.is_file():
        if not cache_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"incomplete feature cache in {run_dir}")
        if json.loads(metadata_path.read_text()) != expected_metadata:
            raise RuntimeError(f"feature cache provenance mismatch in {run_dir}")
        return load_feature_cache(cache_path), expected_metadata

    source_config = yaml.full_load(source_config_path.read_text())
    backbone_config = source_config.get("backbone")
    if not isinstance(backbone_config, ViTConfig):
        raise TypeError(f"source config {source_config_path} does not contain a ViTConfig backbone")
    backbone = backbone_config.instantiate(device=device)
    load_backbone_checkpoint(checkpoint_path, backbone, device)

    train_loader = get_probe_train_dataloader(
        size=backbone_config.img_size,
        batch_size=feature_batch_size,
        root=data_path,
        num_workers=num_workers,
    )
    validation_loader = get_val_dataloader(
        size=backbone_config.img_size,
        batch_size=feature_batch_size,
        root=data_path,
        num_workers=num_workers,
    )
    train_features = extract_dataset_features(backbone, train_loader, device)
    validation_features = extract_dataset_features(backbone, validation_loader, device)
    cached_features = {
        **{f"train_{key}": value for key, value in train_features.items()},
        **{f"validation_{key}": value for key, value in validation_features.items()},
    }
    save_feature_cache(cache_path, cached_features)
    _write_json_atomic(metadata_path, expected_metadata)
    return cached_features, expected_metadata


def _normalize_features(features: torch.Tensor, normalize: bool) -> torch.Tensor:
    return F.layer_norm(features, (features.shape[-1],)) if normalize else features


def _log_validation_curves(
    recipe_name: str,
    training_result: ProbeTrainingResult,
    learning_rates: tuple[float, ...],
    *,
    step_offset: int,
) -> None:
    if len(training_result.validation_curves) != len(learning_rates):
        raise ValueError("probe validation curves must match the learning-rate sweep")
    epochs = len(training_result.validation_curves[0])
    if any(len(curve) != epochs for curve in training_result.validation_curves):
        raise ValueError("probe validation curves must have equal lengths")
    for epoch in range(epochs):
        wandb.log(
            {
                f"probe_calibration/{recipe_name}/lr_{learning_rate:g}/validation_accuracy": (
                    training_result.validation_curves[index][epoch]
                )
                for index, learning_rate in enumerate(learning_rates)
            },
            step=step_offset + epoch,
        )


def _calibrate_source(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_hash: str,
    source: dict[str, Any],
    data_path: Path,
    device: torch.device,
    physical_gpu: int,
) -> None:
    study_id = str(manifest["id"])
    source_id = str(source["id"])
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = (repo_root / str(manifest["log_root"]) / study_id / "runs" / source_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("status") == "completed" and result.get("manifest_sha256") == manifest_hash:
            return
        raise RuntimeError(f"existing result does not match active manifest: {result_path}")

    source_run_dir = (repo_root / str(source["run_dir"])).resolve()
    probe_config = manifest["probe"]
    wandb_config = manifest["wandb"]
    seed = int(probe_config["seed"])
    seed_everything(seed)
    existing_metadata_path = run_dir / "metadata.json"
    existing_metadata = json.loads(existing_metadata_path.read_text()) if existing_metadata_path.is_file() else {}
    started_at = perf_counter()
    initialized_run = wandb.init(
        entity=str(wandb_config["entity"]),
        project=str(wandb_config["project"]),
        group=str(wandb_config["group"]),
        name=source_id,
        id=existing_metadata.get("wandb_run_id"),
        resume="allow" if existing_metadata.get("wandb_run_id") else None,
        mode="online",
        dir=run_dir,
        tags=("probe-calibration", source_id),
        config={
            "study_id": study_id,
            "source": source,
            "recipes": manifest["recipes"],
            "probe": probe_config,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_hash,
            "emitted_data_classes": wandb_config["emitted_data_classes"]["launch"],
        },
    )
    metadata = {
        "study_id": study_id,
        "source_id": source_id,
        "source_run_dir": str(source_run_dir),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "wandb_run_id": initialized_run.id,
        "wandb_url": initialized_run.url,
        "physical_gpu": physical_gpu,
        "hostname": socket.gethostname(),
        "code_shas": {
            "parent": _git_sha(repo_root),
            "mjepa": _git_sha(repo_root.parent / "mjepa"),
            "vit": _git_sha(repo_root.parent / "vit"),
        },
        "tracker": {
            "requested_mode": "online",
            "effective_mode": "online",
            "authorized": True,
            "destination": f"{wandb_config['entity']}/{wandb_config['project']}",
            "emitted_data_classes": wandb_config["emitted_data_classes"]["launch"],
        },
    }
    _write_json_atomic(existing_metadata_path, metadata)

    exit_code = 0
    try:
        features, cache_metadata = _prepare_feature_cache(
            run_dir=run_dir,
            source_run_dir=source_run_dir,
            data_path=data_path,
            device=device,
            feature_batch_size=int(probe_config["feature_batch_size"]),
            num_workers=int(probe_config["num_workers"]),
        )
        recipes: dict[str, Any] = {}
        learning_rates = tuple(float(value) for value in probe_config["learning_rates"])
        train_labels = features["train_labels"]
        validation_labels = features["validation_labels"]
        epochs = int(probe_config["epochs"])
        for recipe_index, recipe in enumerate(manifest["recipes"]):
            recipe_name = str(recipe["id"])
            mode = cast(ProbeFeatureMode, recipe["mode"])
            if mode not in (FINAL_CLS_MODE, LAST_TWO_CLS_MODE):
                raise ValueError(f"unsupported probe feature mode: {mode}")
            train_features = _normalize_features(features[f"train_{mode}"], bool(recipe["normalize"]))
            validation_features = _normalize_features(features[f"validation_{mode}"], bool(recipe["normalize"]))
            recipe_started_at = perf_counter()
            training_result = train_probe_bank(
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                learning_rates=learning_rates,
                epochs=epochs,
                batch_size=int(probe_config["probe_batch_size"]),
                weight_decay=float(probe_config["weight_decay"]),
                warmup_fraction=float(probe_config["warmup_fraction"]),
                start_factor=float(probe_config["start_factor"]),
                final_factor=float(probe_config["final_factor"]),
                device=device,
                seed=seed,
            )
            recipe_result = training_result.to_dict()
            recipe_result["active_seconds"] = perf_counter() - recipe_started_at
            recipes[recipe_name] = recipe_result
            _log_validation_curves(
                recipe_name,
                training_result,
                learning_rates,
                step_offset=recipe_index * epochs,
            )

        best_recipe = max(
            recipes,
            key=lambda recipe_id: (
                float(recipes[recipe_id]["best_peak_accuracy"]),
                -list(recipes).index(recipe_id),
            ),
        )
        best_accuracy = float(recipes[best_recipe]["best_peak_accuracy"])
        online_probe_accuracy = float(source["online_probe_accuracy"])
        result = {
            "status": "completed",
            "study_id": study_id,
            "source_id": source_id,
            "source_role": source["role"],
            "manifest_sha256": manifest_hash,
            "cache": cache_metadata,
            "online_probe_accuracy": online_probe_accuracy,
            "recipes": recipes,
            "best_recipe": best_recipe,
            "best_calibrated_accuracy": best_accuracy,
            "calibration_gain": best_accuracy - online_probe_accuracy,
            "active_seconds": perf_counter() - started_at,
            "wandb_run_id": initialized_run.id,
            "wandb_url": initialized_run.url,
        }
        _write_json_atomic(result_path, result)
        _write_json_atomic(run_dir / "terminal.json", {"status": "completed", "result": str(result_path)})
        initialized_run.summary.update(
            {
                "probe_calibration/best_recipe": best_recipe,
                "probe_calibration/best_accuracy": best_accuracy,
                "probe_calibration/online_accuracy": online_probe_accuracy,
                "probe_calibration/gain": best_accuracy - online_probe_accuracy,
            }
        )
    except Exception as error:
        exit_code = 1
        _write_json_atomic(run_dir / "terminal.json", {"status": "failed", "error": str(error)[:2000]})
        raise
    finally:
        wandb.finish(exit_code=exit_code)


def main(args: Namespace) -> None:
    if args.num_workers <= 0 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker index must be within the configured worker count")
    manifest, manifest_hash = _load_manifest(args.manifest)
    if not args.data.is_dir():
        raise NotADirectoryError(args.data)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    assigned_sources = manifest["sources"][args.worker_index :: args.num_workers]
    for source in assigned_sources:
        _calibrate_source(
            manifest=manifest,
            manifest_path=args.manifest,
            manifest_hash=manifest_hash,
            source=source,
            data_path=args.data,
            device=device,
            physical_gpu=args.physical_gpu,
        )


if __name__ == "__main__":
    main(parse_args())
