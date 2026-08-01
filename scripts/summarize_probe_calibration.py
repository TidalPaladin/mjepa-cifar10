#!/usr/bin/env python3

import hashlib
import json
import os
import tempfile
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Summarize a completed frozen-probe calibration")
    parser.add_argument("manifest", type=Path)
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


def _build_summary(
    manifest: dict[str, Any],
    manifest_hash: str,
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_order = [str(source["id"]) for source in manifest["sources"]]
    if set(results) != set(source_order):
        raise ValueError("result source IDs do not match the calibration manifest")
    for source in manifest["sources"]:
        source_id = str(source["id"])
        result = results[source_id]
        if result.get("status") != "completed" or result.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"incomplete or mismatched result for {source_id}")
        if result.get("source_role") != source["role"]:
            raise ValueError(f"source role mismatch for {source_id}")

    teacher_ids = [source_id for source_id in source_order if results[source_id]["source_role"] == "teacher-baseline"]
    shared_ids = [source_id for source_id in source_order if source_id not in teacher_ids]
    if len(teacher_ids) != 1 or not shared_ids:
        raise ValueError("calibration requires exactly one teacher and at least one shared-student source")

    teacher_id = teacher_ids[0]
    best_shared_id = max(
        shared_ids,
        key=lambda source_id: (
            float(results[source_id]["best_calibrated_accuracy"]),
            -source_order.index(source_id),
        ),
    )
    teacher_accuracy = float(results[teacher_id]["best_calibrated_accuracy"])
    best_shared_accuracy = float(results[best_shared_id]["best_calibrated_accuracy"])
    best_shared_gain = float(results[best_shared_id]["calibration_gain"])
    teacher_gap = teacher_accuracy - best_shared_accuracy
    thresholds = cast(dict[str, Any], manifest["decision"])
    material_probe_lag = best_shared_gain >= float(thresholds["material_probe_lag_gain"])
    below_shared_floor = best_shared_accuracy < float(thresholds["shared_representation_floor"])
    excessive_teacher_gap = teacher_gap > float(thresholds["maximum_teacher_gap_for_probe_explanation"])
    representation_primary = below_shared_floor or excessive_teacher_gap
    if material_probe_lag and representation_primary:
        decision = "material-probe-lag-with-residual-representation-gap"
    elif representation_primary:
        decision = "representation-convergence-primary"
    else:
        decision = "probe-lag-plausibly-explains-gap"

    return {
        "study_id": manifest["id"],
        "manifest_sha256": manifest_hash,
        "updated_at": datetime.now(UTC).isoformat(),
        "teacher_source": teacher_id,
        "teacher_calibrated_accuracy": teacher_accuracy,
        "best_shared_source": best_shared_id,
        "best_shared_calibrated_accuracy": best_shared_accuracy,
        "best_shared_calibration_gain": best_shared_gain,
        "teacher_gap": teacher_gap,
        "material_probe_lag": material_probe_lag,
        "below_shared_representation_floor": below_shared_floor,
        "excessive_teacher_gap": excessive_teacher_gap,
        "representation_convergence_primary": representation_primary,
        "decision": decision,
        "thresholds": thresholds,
        "runs": {source_id: results[source_id] for source_id in source_order},
    }


def main(args: Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("probe calibration manifest must contain a mapping")
    manifest_hash = _sha256(manifest_path)
    repo_root = Path(__file__).resolve().parents[1]
    study_dir = (repo_root / str(manifest["log_root"]) / str(manifest["id"])).resolve()
    results = {
        str(source["id"]): json.loads((study_dir / "runs" / str(source["id"]) / "result.json").read_text())
        for source in manifest["sources"]
    }
    summary = _build_summary(manifest, manifest_hash, results)
    summary_path = study_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
