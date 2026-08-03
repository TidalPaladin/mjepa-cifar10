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
PATCH_MEAN_RANK_KEY: Final[str] = "pretrain/collapse/target_patch_mean/effective_rank_fraction"
PATCH_COSINE_KEY: Final[str] = "pretrain/diversity/target_patch/mean_within_image_pairwise_cosine"
PATCH_ENERGY_KEY: Final[str] = "pretrain/diversity/target_patch/centered_patch_energy_ratio"
VISUAL_IMPROVEMENT_KEY: Final[str] = "pretrain/validation_visual_target_relative_improvement"
PATCH_MEAN_ROUTE: Final[str] = "patch_mean"
BLOCK_TRANSITION_START: Final[int] = 8
LAST_THREE_RECORDS: Final[int] = 3
GATE_TOLERANCE: Final[float] = 1e-12

Aggregation = Literal["min", "max"]
Direction = Literal["minimum", "maximum"]


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Summarize the LeJEPA patch-rank screen")
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


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(values) / len(values)


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


def _safety_gate(
    records: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> tuple[bool, dict[str, float], list[str]]:
    if len(records) < LAST_THREE_RECORDS:
        raise ValueError("safety gate requires at least three validation records")
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
        passed = (
            aggregate >= threshold - GATE_TOLERANCE
            if direction == "minimum"
            else aggregate <= threshold + GATE_TOLERANCE
        )
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
    ordered = sorted(records, key=lambda record: cast(int, record["_step"]))
    if len({record["_step"] for record in ordered}) != len(ordered):
        raise ValueError(f"validation records in {path} must contain unique steps")
    return ordered


def _validate_result(
    result: dict[str, Any],
    source: dict[str, Any],
    manifest_hash: str,
    result_kind: str,
) -> None:
    source_id = str(source["id"])
    if result.get("status") != "completed" or result.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"incomplete or mismatched {result_kind} result for {source_id}")
    if result.get("source_id") != source_id or result.get("source_role") != source["role"]:
        raise ValueError(f"{result_kind} source identity mismatch for {source_id}")


def _layer_by_number(result: dict[str, Any], layer_number: int) -> dict[str, Any]:
    layers = result.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("diagnostic result must contain nonempty layers")
    matches = [layer for layer in layers if isinstance(layer, dict) and layer.get("layer") == layer_number]
    if len(matches) != 1:
        raise ValueError(f"diagnostic result must contain exactly one layer {layer_number}")
    return matches[0]


def _final_layer(result: dict[str, Any]) -> dict[str, Any]:
    layers = result.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(layer, dict) for layer in layers):
        raise ValueError("diagnostic result must contain nonempty layers")
    if not all(isinstance(layer.get("layer"), int) for layer in layers):
        raise ValueError("diagnostic layers must contain integer layer numbers")
    return max(layers, key=lambda layer: cast(int, layer["layer"]))


def _patch_centroid_accuracy(layer: dict[str, Any]) -> float:
    centroid_accuracy = layer.get("centroid_accuracy")
    if not isinstance(centroid_accuracy, dict):
        raise ValueError("diagnostic layer must contain centroid_accuracy")
    value = centroid_accuracy.get(PATCH_MEAN_ROUTE)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("diagnostic patch-mean centroid accuracy must be finite")
    return float(value)


def _build_summary(
    study: dict[str, Any],
    probe_manifest: dict[str, Any],
    diagnostic_manifest: dict[str, Any],
    probe_manifest_hash: str,
    diagnostic_manifest_hash: str,
    records: dict[str, list[dict[str, Any]]],
    probe_results: dict[str, dict[str, Any]],
    diagnostic_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    probe_sources = cast(list[dict[str, Any]], probe_manifest["sources"])
    diagnostic_sources = cast(list[dict[str, Any]], diagnostic_manifest["sources"])
    source_order = [str(source["id"]) for source in probe_sources]
    if [str(source["id"]) for source in diagnostic_sources] != source_order:
        raise ValueError("probe and diagnostic source order must match")
    if set(records) != set(source_order) or set(probe_results) != set(source_order):
        raise ValueError("record and probe-result source IDs must match the probe manifest")
    if set(diagnostic_results) != set(source_order):
        raise ValueError("diagnostic-result source IDs must match the diagnostic manifest")
    controls = [str(source["id"]) for source in probe_sources if source["role"] == "control"]
    if len(controls) != 1:
        raise ValueError("patch-rank screen requires exactly one control")
    if sum(source["role"] == "candidate" for source in probe_sources) < 1:
        raise ValueError("patch-rank screen requires at least one candidate")

    sources = {str(source["id"]): source for source in probe_sources}
    diagnostic_source_map = {str(source["id"]): source for source in diagnostic_sources}
    methodology = cast(dict[str, Any], study["methodology"])
    safety_thresholds = cast(dict[str, Any], methodology["safety_gate"])
    selection_thresholds = cast(dict[str, Any], methodology["terminal_selection_gate"])
    runs: dict[str, dict[str, Any]] = {}
    for source_id in source_order:
        source = sources[source_id]
        if diagnostic_source_map[source_id]["role"] != source["role"]:
            raise ValueError(f"probe and diagnostic roles differ for {source_id}")
        _validate_result(probe_results[source_id], source, probe_manifest_hash, "probe")
        _validate_result(
            diagnostic_results[source_id],
            diagnostic_source_map[source_id],
            diagnostic_manifest_hash,
            "diagnostic",
        )
        source_records = records[source_id]
        terminal_record = source_records[-1]
        safety_passed, safety_observed, safety_violations = _safety_gate(source_records, safety_thresholds)
        block8_patch_accuracy = _patch_centroid_accuracy(
            _layer_by_number(diagnostic_results[source_id], BLOCK_TRANSITION_START)
        )
        final_patch_accuracy = _patch_centroid_accuracy(_final_layer(diagnostic_results[source_id]))
        runs[source_id] = {
            "role": source["role"],
            "objective_complexity_rank": int(source["objective_complexity_rank"]),
            "objective_cost_rank": int(source["objective_cost_rank"]),
            "online_final_accuracy": _metric(terminal_record, VALIDATION_ACCURACY_KEY),
            "online_peak_accuracy": max(_metric(record, VALIDATION_ACCURACY_KEY) for record in source_records),
            "online_step_auc": _curve_auc(source_records, "_step"),
            "online_active_time_auc": _curve_auc(source_records, ACTIVE_SECONDS_KEY),
            "terminal_optimizer_step": int(terminal_record["_step"]),
            "terminal_active_seconds": _metric(terminal_record, ACTIVE_SECONDS_KEY),
            "terminal_patch_mean_rank": _metric(terminal_record, PATCH_MEAN_RANK_KEY),
            "last_three_patch_mean_rank": _mean(
                [_metric(record, PATCH_MEAN_RANK_KEY) for record in source_records[-LAST_THREE_RECORDS:]]
            ),
            "terminal_patch_cosine": _metric(terminal_record, PATCH_COSINE_KEY),
            "terminal_centered_patch_energy": _metric(terminal_record, PATCH_ENERGY_KEY),
            "block8_patch_centroid_accuracy": block8_patch_accuracy,
            "final_patch_centroid_accuracy": final_patch_accuracy,
            "patch_centroid_block8_to_final_gain": final_patch_accuracy - block8_patch_accuracy,
            "frozen_accuracy": float(probe_results[source_id]["best_calibrated_accuracy"]),
            "frozen_recipe": probe_results[source_id]["best_recipe"],
            "probe_calibration_gain": float(probe_results[source_id]["calibration_gain"]),
            "safety_gate_passed": safety_passed,
            "safety_observed": safety_observed,
            "safety_violations": safety_violations,
            "qualifies": False,
            "failure_reasons": [],
        }

    control_id = controls[0]
    control = runs[control_id]
    control_valid = bool(control["safety_gate_passed"])
    if float(control["last_three_patch_mean_rank"]) <= 0:
        raise ValueError("control last-three patch-mean rank must be positive")
    if float(control["terminal_centered_patch_energy"]) <= 0:
        raise ValueError("control centered patch energy must be positive")

    qualifying_sources: list[str] = []
    for source_id in source_order:
        if source_id == control_id:
            continue
        run = runs[source_id]
        rank_ratio = float(run["last_three_patch_mean_rank"]) / float(control["last_three_patch_mean_rank"])
        patch_centroid_gain = float(run["final_patch_centroid_accuracy"]) - float(
            control["final_patch_centroid_accuracy"]
        )
        energy_ratio = float(run["terminal_centered_patch_energy"]) / float(control["terminal_centered_patch_energy"])
        cosine_increase = float(run["terminal_patch_cosine"]) - float(control["terminal_patch_cosine"])
        frozen_accuracy_loss = float(control["frozen_accuracy"]) - float(run["frozen_accuracy"])
        step_auc_loss = float(control["online_step_auc"]) - float(run["online_step_auc"])
        rank_passed = rank_ratio >= (
            float(selection_thresholds["last_three_patch_mean_rank_ratio_minimum"]) - GATE_TOLERANCE
        )
        patch_centroid_passed = patch_centroid_gain >= (
            float(selection_thresholds["final_patch_centroid_accuracy_gain_minimum"]) - GATE_TOLERANCE
        )
        energy_passed = energy_ratio >= (
            float(selection_thresholds["candidate_to_control_centered_patch_energy_ratio_minimum"]) - GATE_TOLERANCE
        )
        cosine_passed = cosine_increase <= (
            float(selection_thresholds["candidate_patch_cosine_maximum_increase"]) + GATE_TOLERANCE
        )
        frozen_accuracy_passed = frozen_accuracy_loss <= (
            float(selection_thresholds["frozen_accuracy_loss_maximum"]) + GATE_TOLERANCE
        )
        step_auc_passed = step_auc_loss <= (
            float(selection_thresholds["online_step_auc_loss_maximum"]) + GATE_TOLERANCE
        )
        failure_reasons: list[str] = []
        if not control_valid:
            failure_reasons.append("control-safety-gate")
        if not run["safety_gate_passed"]:
            failure_reasons.append("safety-gate")
        if not rank_passed:
            failure_reasons.append("patch-mean-rank")
        if not patch_centroid_passed:
            failure_reasons.append("patch-centroid-accuracy")
        if not energy_passed:
            failure_reasons.append("centered-patch-energy")
        if not cosine_passed:
            failure_reasons.append("patch-cosine")
        if not frozen_accuracy_passed:
            failure_reasons.append("frozen-accuracy")
        if not step_auc_passed:
            failure_reasons.append("online-step-auc")
        qualifies = not failure_reasons
        run.update(
            {
                "last_three_patch_mean_rank_ratio": rank_ratio,
                "final_patch_centroid_accuracy_gain": patch_centroid_gain,
                "candidate_to_control_centered_patch_energy_ratio": energy_ratio,
                "candidate_patch_cosine_increase": cosine_increase,
                "frozen_accuracy_loss": frozen_accuracy_loss,
                "online_step_auc_loss": step_auc_loss,
                "online_active_time_auc_loss": float(control["online_active_time_auc"])
                - float(run["online_active_time_auc"]),
                "patch_mean_rank_gate_passed": rank_passed,
                "patch_centroid_gate_passed": patch_centroid_passed,
                "centered_patch_energy_gate_passed": energy_passed,
                "patch_cosine_gate_passed": cosine_passed,
                "frozen_accuracy_gate_passed": frozen_accuracy_passed,
                "online_step_auc_gate_passed": step_auc_passed,
                "qualifies": qualifies,
                "failure_reasons": failure_reasons,
            }
        )
        if qualifies:
            qualifying_sources.append(source_id)

    qualifying_sources.sort(
        key=lambda source_id: (
            -float(runs[source_id]["frozen_accuracy"]),
            -float(runs[source_id]["final_patch_centroid_accuracy"]),
            -float(runs[source_id]["last_three_patch_mean_rank_ratio"]),
            -float(runs[source_id]["online_step_auc"]),
            -float(runs[source_id]["online_active_time_auc"]),
            int(runs[source_id]["objective_complexity_rank"]),
            int(runs[source_id]["objective_cost_rank"]),
            source_order.index(source_id),
        )
    )
    selected_source = qualifying_sources[0] if qualifying_sources else None
    if not control_valid:
        decision = "control-failed-preregistered-safety-gate"
    elif selected_source is None:
        decision = "no-candidate-passed-preregistered-gates"
    else:
        decision = "candidate-selected-for-preregistered-long-horizon"
    return {
        "study_id": study["id"],
        "probe_study_id": probe_manifest["id"],
        "diagnostic_study_id": diagnostic_manifest["id"],
        "probe_manifest_sha256": probe_manifest_hash,
        "diagnostic_manifest_sha256": diagnostic_manifest_hash,
        "updated_at": datetime.now(UTC).isoformat(),
        "control_source": control_id,
        "qualifying_sources": qualifying_sources,
        "selected_source": selected_source,
        "decision": decision,
        "safety_thresholds": safety_thresholds,
        "selection_thresholds": selection_thresholds,
        "last_three_patch_mean_rank_definition": "arithmetic mean over the final three scheduled validation records",
        "auc_definition": "trapezoidal mean accuracy between the first and final scheduled validation records",
        "block_transition_definition": (
            f"block-{BLOCK_TRANSITION_START}-to-final normalized patch-mean centroid accuracy"
        ),
        "runs": runs,
    }


def main(args: Namespace) -> None:
    probe_manifest_path = args.manifest.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    probe_manifest = yaml.safe_load(probe_manifest_path.read_text())
    if not isinstance(probe_manifest, dict):
        raise TypeError("probe calibration manifest must contain a mapping")
    probe_manifest_hash = _sha256(probe_manifest_path)
    diagnostic_manifest_path = (repo_root / str(probe_manifest["diagnostic_manifest"])).resolve()
    diagnostic_manifest = yaml.safe_load(diagnostic_manifest_path.read_text())
    if not isinstance(diagnostic_manifest, dict):
        raise TypeError("diagnostic manifest must contain a mapping")
    diagnostic_manifest_hash = _sha256(diagnostic_manifest_path)
    study_path = (repo_root / str(probe_manifest["screen_study"])).resolve()
    study = yaml.safe_load(study_path.read_text())
    if not isinstance(study, dict):
        raise TypeError("screen study must contain a mapping")

    log_root = (repo_root / str(probe_manifest["log_root"])).resolve()
    probe_study_dir = log_root / str(probe_manifest["id"])
    diagnostic_study_dir = log_root / str(diagnostic_manifest["id"])
    records: dict[str, list[dict[str, Any]]] = {}
    probe_results: dict[str, dict[str, Any]] = {}
    diagnostic_results: dict[str, dict[str, Any]] = {}
    for source in probe_manifest["sources"]:
        source_id = str(source["id"])
        source_run_dir = (repo_root / str(source["run_dir"])).resolve()
        records[source_id] = _read_validation_records(source_run_dir / "metrics.jsonl")
        probe_results[source_id] = json.loads((probe_study_dir / "runs" / source_id / "result.json").read_text())
        diagnostic_results[source_id] = json.loads(
            (diagnostic_study_dir / "runs" / source_id / "result.json").read_text()
        )

    summary = _build_summary(
        study,
        probe_manifest,
        diagnostic_manifest,
        probe_manifest_hash,
        diagnostic_manifest_hash,
        records,
        probe_results,
        diagnostic_results,
    )
    summary.update(
        {
            "probe_manifest": str(probe_manifest_path),
            "diagnostic_manifest": str(diagnostic_manifest_path),
            "screen_study": str(study_path),
            "screen_study_sha256": _sha256(study_path),
        }
    )
    summary_path = probe_study_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
