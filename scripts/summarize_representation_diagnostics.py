#!/usr/bin/env python3

import hashlib
import json
import os
import tempfile
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Summarize completed layerwise representation diagnostics")
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
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


def _build_summary(
    manifest: dict[str, Any],
    manifest_hash: str,
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_order = [str(source["id"]) for source in manifest["sources"]]
    if set(results) != set(source_order):
        raise ValueError("result source IDs do not match the diagnostic manifest")
    for source in manifest["sources"]:
        source_id = str(source["id"])
        result = results[source_id]
        if result.get("status") != "completed" or result.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"incomplete or mismatched result for {source_id}")
        if result.get("source_role") != source["role"]:
            raise ValueError(f"source role mismatch for {source_id}")

    teacher_ids = [source_id for source_id in source_order if results[source_id]["source_role"] == "teacher-baseline"]
    if not teacher_ids:
        return _build_control_candidate_summary(manifest, manifest_hash, results, source_order)

    candidate_ids = [source_id for source_id in source_order if source_id not in teacher_ids]
    if len(teacher_ids) != 1 or len(candidate_ids) != 1:
        raise ValueError("representation diagnostics require exactly one teacher and one candidate")
    teacher_id = teacher_ids[0]
    candidate_id = candidate_ids[0]
    teacher_layers = results[teacher_id]["layers"]
    candidate_layers = results[candidate_id]["layers"]
    if len(teacher_layers) != len(candidate_layers) or not teacher_layers:
        raise ValueError("teacher and candidate must report the same nonempty layer count")

    layers: list[dict[str, Any]] = []
    for teacher_layer, candidate_layer in zip(teacher_layers, candidate_layers, strict=True):
        if teacher_layer["layer"] != candidate_layer["layer"]:
            raise ValueError("teacher and candidate layer indices do not align")
        centroid_gaps = {
            route: float(teacher_layer["centroid_accuracy"][route]) - float(candidate_layer["centroid_accuracy"][route])
            for route in ("cls", "patch_mean", "cls_patch_mean")
        }
        teacher_energy = float(teacher_layer["patch_diversity"]["centered_patch_energy_ratio"])
        if teacher_energy <= 0:
            raise ValueError("teacher centered patch energy must be positive")
        layers.append(
            {
                "layer": int(teacher_layer["layer"]),
                "centroid_accuracy_gap": centroid_gaps,
                "teacher_centroid_accuracy": teacher_layer["centroid_accuracy"],
                "candidate_centroid_accuracy": candidate_layer["centroid_accuracy"],
                "candidate_cpa_excess": float(candidate_layer["cls_patch_alignment"]["cpa_mean"])
                - float(teacher_layer["cls_patch_alignment"]["cpa_mean"]),
                "candidate_patch_pair_cosine_excess": float(
                    candidate_layer["patch_diversity"]["mean_within_image_pairwise_cosine"]
                )
                - float(teacher_layer["patch_diversity"]["mean_within_image_pairwise_cosine"]),
                "candidate_to_teacher_centered_energy_ratio": float(
                    candidate_layer["patch_diversity"]["centered_patch_energy_ratio"]
                )
                / teacher_energy,
            }
        )

    thresholds = manifest["decision"]
    final = layers[-1]
    spatial_homogenization_supported = (
        final["candidate_cpa_excess"] >= float(thresholds["minimum_candidate_cpa_excess"])
        and final["candidate_patch_pair_cosine_excess"]
        >= float(thresholds["minimum_candidate_patch_pair_cosine_excess"])
        and final["candidate_to_teacher_centered_energy_ratio"]
        <= float(thresholds["maximum_candidate_to_teacher_centered_energy_ratio"])
    )
    material_gap = float(thresholds["material_centroid_accuracy_gap"])
    representation_gap_supported = (
        final["centroid_accuracy_gap"]["cls"] >= material_gap
        and final["centroid_accuracy_gap"]["cls_patch_mean"] >= material_gap
    )
    first_material_cls_gap_layer = next(
        (layer["layer"] for layer in layers if layer["centroid_accuracy_gap"]["cls"] >= material_gap),
        None,
    )
    if spatial_homogenization_supported and representation_gap_supported:
        decision = "spatial-homogenization-with-representation-gap"
    elif spatial_homogenization_supported:
        decision = "spatial-homogenization-without-material-centroid-gap"
    elif representation_gap_supported:
        decision = "representation-gap-without-spatial-homogenization"
    else:
        decision = "neither-preregistered-bottleneck-supported"

    return {
        "study_id": manifest["id"],
        "manifest_sha256": manifest_hash,
        "updated_at": datetime.now(UTC).isoformat(),
        "teacher_source": teacher_id,
        "candidate_source": candidate_id,
        "spatial_homogenization_supported": spatial_homogenization_supported,
        "representation_gap_supported": representation_gap_supported,
        "first_material_cls_gap_layer": first_material_cls_gap_layer,
        "decision": decision,
        "thresholds": thresholds,
        "layers": layers,
        "runs": {source_id: results[source_id] for source_id in source_order},
    }


def _build_control_candidate_summary(
    manifest: dict[str, Any],
    manifest_hash: str,
    results: dict[str, dict[str, Any]],
    source_order: list[str],
) -> dict[str, Any]:
    control_ids = [source_id for source_id in source_order if results[source_id]["source_role"] == "control"]
    candidate_ids = [source_id for source_id in source_order if results[source_id]["source_role"] == "candidate"]
    if len(control_ids) != 1 or len(candidate_ids) != 1 or len(source_order) != 2:
        raise ValueError("representation diagnostics require either teacher/candidate or control/candidate sources")

    control_id = control_ids[0]
    candidate_id = candidate_ids[0]
    control_layers = results[control_id]["layers"]
    candidate_layers = results[candidate_id]["layers"]
    if len(control_layers) != len(candidate_layers) or not control_layers:
        raise ValueError("control and candidate must report the same nonempty layer count")

    layers: list[dict[str, Any]] = []
    for control_layer, candidate_layer in zip(control_layers, candidate_layers, strict=True):
        if control_layer["layer"] != candidate_layer["layer"]:
            raise ValueError("control and candidate layer indices do not align")
        control_energy = float(control_layer["patch_diversity"]["centered_patch_energy_ratio"])
        if control_energy <= 0:
            raise ValueError("control centered patch energy must be positive")
        layers.append(
            {
                "layer": int(control_layer["layer"]),
                "centroid_accuracy_gain": {
                    route: float(candidate_layer["centroid_accuracy"][route])
                    - float(control_layer["centroid_accuracy"][route])
                    for route in ("cls", "patch_mean", "cls_patch_mean")
                },
                "control_centroid_accuracy": control_layer["centroid_accuracy"],
                "candidate_centroid_accuracy": candidate_layer["centroid_accuracy"],
                "candidate_cpa_excess": float(candidate_layer["cls_patch_alignment"]["cpa_mean"])
                - float(control_layer["cls_patch_alignment"]["cpa_mean"]),
                "candidate_patch_pair_cosine_excess": float(
                    candidate_layer["patch_diversity"]["mean_within_image_pairwise_cosine"]
                )
                - float(control_layer["patch_diversity"]["mean_within_image_pairwise_cosine"]),
                "candidate_to_control_centered_energy_ratio": float(
                    candidate_layer["patch_diversity"]["centered_patch_energy_ratio"]
                )
                / control_energy,
            }
        )

    final_patch_gain = float(layers[-1]["centroid_accuracy_gain"]["patch_mean"])
    return {
        "study_id": manifest["id"],
        "manifest_sha256": manifest_hash,
        "updated_at": datetime.now(UTC).isoformat(),
        "control_source": control_id,
        "candidate_source": candidate_id,
        "decision": (
            "candidate-higher-final-patch-mean"
            if final_patch_gain > 0.0
            else "control-higher-or-equal-final-patch-mean"
        ),
        "thresholds": manifest["decision"],
        "layers": layers,
        "runs": {source_id: results[source_id] for source_id in source_order},
    }


def main(args: Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("representation diagnostic manifest must contain a mapping")
    manifest_hash = _sha256(manifest_path)
    repository = Path(__file__).resolve().parents[1]
    study_dir = repository / str(manifest["log_root"]) / str(manifest["id"])
    results = {
        str(source["id"]): json.loads((study_dir / "runs" / str(source["id"]) / "result.json").read_text())
        for source in manifest["sources"]
    }
    summary = _build_summary(manifest, manifest_hash, results)
    _write_json_atomic(study_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
