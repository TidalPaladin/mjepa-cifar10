#!/usr/bin/env python3

import hashlib
import json
import math
import os
import tempfile
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml


VALIDATION_ACCURACY_KEY: Final[str] = "probe/validation_accuracy"
ACTIVE_SECONDS_KEY: Final[str] = "convergence/active_seconds"
PATCH_COSINE_KEY: Final[str] = "pretrain/diversity/target_patch/mean_within_image_pairwise_cosine"
PATCH_ENERGY_KEY: Final[str] = "pretrain/diversity/target_patch/centered_patch_energy_ratio"
PATCH_RANK_KEY: Final[str] = "pretrain/diversity/target_patch/centered_patch_effective_rank_fraction"
CPA_KEY: Final[str] = "pretrain/validation/cpa_mean"
VISUAL_IMPROVEMENT_KEY: Final[str] = "pretrain/validation_visual_target_relative_improvement"
LAST_THREE_RECORDS: Final[int] = 3

Aggregation = Literal["min", "max"]
Direction = Literal["minimum", "maximum"]


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Summarize the LeJEPA token-diversity objective screen")
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


def _metric(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"validation record is missing numeric metric {key!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"validation metric {key!r} must be finite")
    return result


def _curve_auc(records: list[dict[str, Any]], x_key: str) -> float:
    if len(records) < 2:
        raise ValueError("AUC requires at least two validation records")
    coordinates = [
        (
            float(record["_step"]) if x_key == "_step" else _metric(record, x_key),
            _metric(record, VALIDATION_ACCURACY_KEY),
        )
        for record in records
    ]
    if any(right[0] <= left[0] for left, right in zip(coordinates, coordinates[1:], strict=False)):
        raise ValueError(f"validation coordinate {x_key!r} must increase strictly")
    width = coordinates[-1][0] - coordinates[0][0]
    area = math.fsum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(coordinates, coordinates[1:], strict=False)
    )
    return area / width


def _collapse_gate(
    records: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> tuple[bool, dict[str, float], list[str]]:
    if len(records) < LAST_THREE_RECORDS:
        raise ValueError("collapse gate requires at least three validation records")
    terminal_records = records[-LAST_THREE_RECORDS:]
    finite_required = float(thresholds["finite_fraction_required"])
    checks: tuple[tuple[str, str, Aggregation, float, Direction], ...] = (
        (
            "target_cls_finite_fraction",
            "pretrain/collapse/target_cls/finite_fraction",
            "min",
            finite_required,
            "minimum",
        ),
        (
            "target_patch_mean_finite_fraction",
            "pretrain/collapse/target_patch_mean/finite_fraction",
            "min",
            finite_required,
            "minimum",
        ),
        (
            "target_patch_finite_image_fraction",
            "pretrain/diversity/target_patch/finite_image_fraction",
            "min",
            finite_required,
            "minimum",
        ),
        (
            "target_cls_std_mean",
            "pretrain/collapse/target_cls/std_mean",
            "min",
            float(thresholds["target_cls_std_mean_minimum"]),
            "minimum",
        ),
        (
            "target_cls_effective_rank_fraction",
            "pretrain/collapse/target_cls/effective_rank_fraction",
            "min",
            float(thresholds["target_cls_effective_rank_fraction_minimum"]),
            "minimum",
        ),
        (
            "target_cls_top_eigenvalue_fraction",
            "pretrain/collapse/target_cls/top_eigenvalue_fraction",
            "max",
            float(thresholds["target_cls_top_eigenvalue_fraction_maximum"]),
            "maximum",
        ),
        (
            "target_cls_mean_pairwise_cosine",
            "pretrain/collapse/target_cls/mean_pairwise_cosine",
            "max",
            float(thresholds["target_cls_mean_pairwise_cosine_maximum"]),
            "maximum",
        ),
        (
            "target_patch_mean_std_mean",
            "pretrain/collapse/target_patch_mean/std_mean",
            "min",
            float(thresholds["target_patch_mean_std_mean_minimum"]),
            "minimum",
        ),
        (
            "target_patch_mean_effective_rank_fraction",
            "pretrain/collapse/target_patch_mean/effective_rank_fraction",
            "min",
            float(thresholds["target_patch_mean_effective_rank_fraction_minimum"]),
            "minimum",
        ),
        (
            "target_patch_mean_top_eigenvalue_fraction",
            "pretrain/collapse/target_patch_mean/top_eigenvalue_fraction",
            "max",
            float(thresholds["target_patch_mean_top_eigenvalue_fraction_maximum"]),
            "maximum",
        ),
        (
            "target_patch_mean_mean_pairwise_cosine",
            "pretrain/collapse/target_patch_mean/mean_pairwise_cosine",
            "max",
            float(thresholds["target_patch_mean_mean_pairwise_cosine_maximum"]),
            "maximum",
        ),
        (
            "visual_target_relative_improvement",
            VISUAL_IMPROVEMENT_KEY,
            "min",
            float(thresholds["visual_target_relative_improvement_minimum"]),
            "minimum",
        ),
    )
    observed: dict[str, float] = {}
    violations: list[str] = []
    for name, metric_key, aggregation, threshold, direction in checks:
        values = [_metric(record, metric_key) for record in terminal_records]
        aggregate = min(values) if aggregation == "min" else max(values)
        observed[name] = aggregate
        passed = aggregate >= threshold if direction == "minimum" else aggregate <= threshold
        if not passed:
            violations.append(f"{name}:{aggregate:.12g}:{direction}:{threshold:.12g}")
    return not violations, observed, violations


def _read_validation_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as input_file:
        for line in input_file:
            record = json.loads(line)
            if VALIDATION_ACCURACY_KEY in record:
                records.append(record)
    if len(records) < LAST_THREE_RECORDS:
        raise ValueError(f"expected at least three validation records in {path}")
    if any(not isinstance(record.get("_step"), int) for record in records):
        raise ValueError(f"validation records in {path} must contain integer steps")
    return sorted(records, key=lambda record: cast(int, record["_step"]))


def _validate_inputs(
    manifest: dict[str, Any],
    manifest_hash: str,
    records: dict[str, list[dict[str, Any]]],
    results: dict[str, dict[str, Any]],
) -> tuple[list[str], str]:
    sources = cast(list[dict[str, Any]], manifest["sources"])
    source_order = [str(source["id"]) for source in sources]
    if set(records) != set(source_order) or set(results) != set(source_order):
        raise ValueError("record and result source IDs must match the probe manifest")
    controls = [str(source["id"]) for source in sources if source["role"] == "control"]
    if len(controls) != 1:
        raise ValueError("token-diversity screen requires exactly one control")
    if sum(source["role"] == "candidate" for source in sources) < 1:
        raise ValueError("token-diversity screen requires at least one candidate")
    for source in sources:
        source_id = str(source["id"])
        result = results[source_id]
        if result.get("status") != "completed" or result.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"incomplete or mismatched probe result for {source_id}")
        if result.get("source_role") != source["role"]:
            raise ValueError(f"source role mismatch for {source_id}")
    return source_order, controls[0]


def _build_summary(
    study: dict[str, Any],
    manifest: dict[str, Any],
    manifest_hash: str,
    records: dict[str, list[dict[str, Any]]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_order, control_id = _validate_inputs(manifest, manifest_hash, records, results)
    sources = {str(source["id"]): source for source in cast(list[dict[str, Any]], manifest["sources"])}
    methodology = cast(dict[str, Any], study["methodology"])
    collapse_thresholds = cast(dict[str, Any], methodology["last_three_validation_gate"])
    selection_thresholds = cast(dict[str, Any], methodology["terminal_selection_gate"])
    runs: dict[str, dict[str, Any]] = {}
    for source_id in source_order:
        source_records = records[source_id]
        terminal_record = source_records[-1]
        collapse_passed, collapse_observed, collapse_violations = _collapse_gate(
            source_records,
            collapse_thresholds,
        )
        runs[source_id] = {
            "role": sources[source_id]["role"],
            "online_final_accuracy": _metric(terminal_record, VALIDATION_ACCURACY_KEY),
            "online_peak_accuracy": max(_metric(record, VALIDATION_ACCURACY_KEY) for record in source_records),
            "online_step_auc": _curve_auc(source_records, "_step"),
            "online_active_time_auc": _curve_auc(source_records, ACTIVE_SECONDS_KEY),
            "terminal_optimizer_step": int(terminal_record["_step"]),
            "terminal_active_seconds": _metric(terminal_record, ACTIVE_SECONDS_KEY),
            "terminal_patch_cosine": _metric(terminal_record, PATCH_COSINE_KEY),
            "terminal_centered_patch_energy": _metric(terminal_record, PATCH_ENERGY_KEY),
            "terminal_centered_patch_rank": _metric(terminal_record, PATCH_RANK_KEY),
            "terminal_cls_patch_alignment": _metric(terminal_record, CPA_KEY),
            "collapse_gate_passed": collapse_passed,
            "collapse_observed": collapse_observed,
            "collapse_violations": collapse_violations,
            "frozen_accuracy": float(results[source_id]["best_calibrated_accuracy"]),
            "frozen_recipe": results[source_id]["best_recipe"],
            "probe_calibration_gain": float(results[source_id]["calibration_gain"]),
            "probe_result": results[source_id],
            "qualifies": False,
            "failure_reasons": [],
        }

    control = runs[control_id]
    qualifying_sources: list[str] = []
    energy_ratio_minimum = float(selection_thresholds["candidate_to_control_centered_patch_energy_ratio_minimum"])
    cosine_improvement_minimum = float(selection_thresholds["candidate_patch_pair_cosine_improvement_minimum"])
    frozen_gain_minimum = float(selection_thresholds["frozen_accuracy_gain_minimum"])
    frozen_loss_maximum = float(selection_thresholds["frozen_equivalence_accuracy_loss_maximum"])
    step_auc_gain_minimum = float(selection_thresholds["online_step_auc_gain_if_frozen_equivalent"])
    for source_id in source_order:
        if source_id == control_id:
            continue
        run = runs[source_id]
        energy_ratio = float(run["terminal_centered_patch_energy"]) / float(control["terminal_centered_patch_energy"])
        cosine_improvement = float(control["terminal_patch_cosine"]) - float(run["terminal_patch_cosine"])
        frozen_gain = float(run["frozen_accuracy"]) - float(control["frozen_accuracy"])
        step_auc_gain = float(run["online_step_auc"]) / float(control["online_step_auc"]) - 1.0
        active_time_auc_gain = float(run["online_active_time_auc"]) / float(control["online_active_time_auc"]) - 1.0
        spatial_passed = energy_ratio >= energy_ratio_minimum and cosine_improvement >= cosine_improvement_minimum
        frozen_passed = frozen_gain >= frozen_gain_minimum or (
            frozen_gain >= -frozen_loss_maximum and step_auc_gain >= step_auc_gain_minimum
        )
        failure_reasons: list[str] = []
        if not run["collapse_gate_passed"]:
            failure_reasons.append("collapse-gate")
        if energy_ratio < energy_ratio_minimum:
            failure_reasons.append("centered-patch-energy")
        if cosine_improvement < cosine_improvement_minimum:
            failure_reasons.append("patch-cosine")
        if not frozen_passed:
            failure_reasons.append("frozen-accuracy-or-equivalent-auc")
        qualifies = not failure_reasons
        run.update(
            {
                "candidate_to_control_centered_patch_energy_ratio": energy_ratio,
                "candidate_patch_cosine_improvement": cosine_improvement,
                "frozen_accuracy_gain": frozen_gain,
                "online_step_auc_gain": step_auc_gain,
                "online_active_time_auc_gain": active_time_auc_gain,
                "spatial_gate_passed": spatial_passed,
                "frozen_gate_passed": frozen_passed,
                "qualifies": qualifies,
                "failure_reasons": failure_reasons,
            }
        )
        if qualifies:
            qualifying_sources.append(source_id)

    qualifying_sources.sort(
        key=lambda source_id: (
            -float(runs[source_id]["frozen_accuracy"]),
            -float(runs[source_id]["online_step_auc"]),
            -float(runs[source_id]["online_active_time_auc"]),
            int(sources[source_id]["objective_complexity_rank"]),
            source_order.index(source_id),
        )
    )
    selected_source = qualifying_sources[0] if qualifying_sources else None
    decision = (
        "candidate-selected-for-preregistered-long-horizon"
        if selected_source is not None
        else "no-candidate-passed-preregistered-gates"
    )
    return {
        "study_id": study["id"],
        "probe_study_id": manifest["id"],
        "manifest_sha256": manifest_hash,
        "updated_at": datetime.now(UTC).isoformat(),
        "control_source": control_id,
        "qualifying_sources": qualifying_sources,
        "selected_source": selected_source,
        "decision": decision,
        "collapse_thresholds": collapse_thresholds,
        "selection_thresholds": selection_thresholds,
        "auc_definition": "trapezoidal mean accuracy between the first and final scheduled validation records",
        "runs": runs,
    }


def main(args: Namespace) -> None:
    manifest_path = args.manifest.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("probe calibration manifest must contain a mapping")
    manifest_hash = _sha256(manifest_path)
    study_path = (repo_root / str(manifest["screen_study"])).resolve()
    study = yaml.safe_load(study_path.read_text())
    if not isinstance(study, dict):
        raise TypeError("screen study must contain a mapping")
    study_dir = (repo_root / str(manifest["log_root"]) / str(manifest["id"])).resolve()
    records: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, dict[str, Any]] = {}
    for source in manifest["sources"]:
        source_id = str(source["id"])
        source_run_dir = (repo_root / str(source["run_dir"])).resolve()
        records[source_id] = _read_validation_records(source_run_dir / "metrics.jsonl")
        result_path = study_dir / "runs" / source_id / "result.json"
        results[source_id] = json.loads(result_path.read_text())
    summary = _build_summary(study, manifest, manifest_hash, records, results)
    summary.update(
        {
            "manifest": str(manifest_path),
            "screen_study": str(study_path),
            "screen_study_sha256": _sha256(study_path),
        }
    )
    summary_path = study_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
