from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Final, Literal, Sequence

from .models import PromotionRules


TARGET_FRACTIONS: Final[tuple[float, float]] = (0.90, 0.95)
Axis = Literal["step", "active_seconds"]


@dataclass(frozen=True)
class MetricPoint:
    step: int
    active_seconds: float
    accuracy: float


@dataclass(frozen=True)
class ConvergenceSummary:
    peak_accuracy: float
    final_accuracy: float
    step_to_90: int | None
    step_to_95: int | None
    active_seconds_to_90: float | None
    active_seconds_to_95: float | None
    step_auc: float
    active_time_auc: float
    step_horizon: int
    active_time_horizon: float

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    criterion: Literal["accuracy", "time_to_95", "time_auc"] | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ConfirmationDecision:
    confirmed: bool
    criterion: str | None
    paired_improvements: int
    baseline_mean: float
    baseline_std: float
    candidate_mean: float
    candidate_std: float
    mean_paired_difference: float
    reasons: tuple[str, ...]


def derive_convergence_targets(baseline_peak_accuracy: float) -> tuple[float, float]:
    if not 0.0 <= baseline_peak_accuracy <= 1.0:
        raise ValueError("baseline peak accuracy must be between 0 and 1")
    return tuple(fraction * baseline_peak_accuracy for fraction in TARGET_FRACTIONS)  # type: ignore[return-value]


def _validate_points(points: Sequence[MetricPoint]) -> tuple[MetricPoint, ...]:
    if not points:
        raise ValueError("at least one metric point is required")
    ordered = tuple(sorted(points, key=lambda point: (point.step, point.active_seconds)))
    if any(not 0.0 <= point.accuracy <= 1.0 for point in ordered):
        raise ValueError("accuracies must be between 0 and 1")
    if any(point.step < 0 or point.active_seconds < 0 for point in ordered):
        raise ValueError("steps and active seconds must be nonnegative")
    if any(current.step <= previous.step for previous, current in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("metric steps must be strictly increasing")
    if any(
        current.active_seconds <= previous.active_seconds
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("active seconds must be strictly increasing")
    return ordered


def _first_target(points: Sequence[MetricPoint], target: float, axis: Axis) -> int | float | None:
    for point in points:
        if point.accuracy >= target:
            return point.step if axis == "step" else point.active_seconds
    return None


def _accuracy_at(points: Sequence[MetricPoint], axis: Axis, position: float) -> float:
    first_position = float(getattr(points[0], axis))
    if position <= first_position:
        return points[0].accuracy
    for previous, current in zip(points, points[1:], strict=False):
        previous_position = float(getattr(previous, axis))
        current_position = float(getattr(current, axis))
        if position <= current_position:
            fraction = (position - previous_position) / (current_position - previous_position)
            return previous.accuracy + fraction * (current.accuracy - previous.accuracy)
    return points[-1].accuracy


def accuracy_auc(points: Sequence[MetricPoint], axis: Axis, horizon: float) -> float:
    """Return trapezoidal accuracy AUC normalized to the supplied common horizon."""
    ordered = _validate_points(points)
    if horizon <= 0:
        raise ValueError("AUC horizon must be positive")
    coordinates: list[tuple[float, float]] = [(0.0, ordered[0].accuracy)]
    coordinates.extend(
        (float(getattr(point, axis)), point.accuracy) for point in ordered if 0 < float(getattr(point, axis)) < horizon
    )
    coordinates.append((horizon, _accuracy_at(ordered, axis, horizon)))
    coordinates = sorted(set(coordinates))
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(coordinates, coordinates[1:], strict=False)
    )
    return area / horizon


def summarize_convergence(
    points: Sequence[MetricPoint],
    baseline_peak_accuracy: float,
    *,
    step_horizon: int | None = None,
    active_time_horizon: float | None = None,
) -> ConvergenceSummary:
    ordered = _validate_points(points)
    target_90, target_95 = derive_convergence_targets(baseline_peak_accuracy)
    effective_step_horizon = step_horizon if step_horizon is not None else ordered[-1].step
    effective_time_horizon = active_time_horizon if active_time_horizon is not None else ordered[-1].active_seconds
    if effective_step_horizon <= 0 or effective_time_horizon <= 0:
        raise ValueError("convergence horizons must be positive")
    return ConvergenceSummary(
        peak_accuracy=max(point.accuracy for point in ordered),
        final_accuracy=ordered[-1].accuracy,
        step_to_90=_as_int_or_none(_first_target(ordered, target_90, "step")),
        step_to_95=_as_int_or_none(_first_target(ordered, target_95, "step")),
        active_seconds_to_90=_as_float_or_none(_first_target(ordered, target_90, "active_seconds")),
        active_seconds_to_95=_as_float_or_none(_first_target(ordered, target_95, "active_seconds")),
        step_auc=accuracy_auc(ordered, "step", effective_step_horizon),
        active_time_auc=accuracy_auc(ordered, "active_seconds", effective_time_horizon),
        step_horizon=effective_step_horizon,
        active_time_horizon=effective_time_horizon,
    )


def _as_int_or_none(value: int | float | None) -> int | None:
    return int(value) if value is not None else None


def _as_float_or_none(value: int | float | None) -> float | None:
    return float(value) if value is not None else None


def promotion_decision(
    baseline: ConvergenceSummary,
    candidate: ConvergenceSummary,
    rules: PromotionRules | None = None,
) -> PromotionDecision:
    rules = rules or PromotionRules()
    reasons: list[str] = []
    accuracy_floor = baseline.peak_accuracy - rules.maximum_accuracy_loss
    accuracy_qualifies = candidate.peak_accuracy >= baseline.peak_accuracy + rules.accuracy_gain
    if accuracy_qualifies:
        reasons.append("peak probe accuracy improved by the required margin")

    convergence_qualifies = (
        baseline.active_seconds_to_95 is not None
        and candidate.active_seconds_to_95 is not None
        and candidate.active_seconds_to_95 <= baseline.active_seconds_to_95 * (1 - rules.convergence_gain)
        and candidate.peak_accuracy >= accuracy_floor
    )
    if convergence_qualifies:
        reasons.append("active time to the fixed 95% target improved without excessive peak-accuracy loss")

    auc_qualifies = (
        candidate.active_time_auc >= baseline.active_time_auc * (1 + rules.auc_gain)
        and candidate.peak_accuracy >= accuracy_floor
    )
    if auc_qualifies:
        reasons.append("common-horizon active-time AUC improved without excessive peak-accuracy loss")

    criterion = None
    if accuracy_qualifies:
        criterion = "accuracy"
    elif convergence_qualifies:
        criterion = "time_to_95"
    elif auc_qualifies:
        criterion = "time_auc"
    if criterion is None:
        reasons.append("candidate did not meet any promotion threshold")
    return PromotionDecision(criterion is not None, criterion, tuple(reasons))


def rank_promoted_candidates(
    candidates: Sequence[tuple[str, ConvergenceSummary]],
) -> tuple[tuple[str, ConvergenceSummary], ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item[1].active_time_auc,
                -item[1].peak_accuracy,
                item[1].active_seconds_to_95 if item[1].active_seconds_to_95 is not None else math.inf,
            ),
        )
    )


def confirmation_decision(
    baseline_by_seed: Sequence[ConvergenceSummary],
    candidate_by_seed: Sequence[ConvergenceSummary],
    criterion: Literal["accuracy", "time_to_95", "time_auc"],
    rules: PromotionRules | None = None,
) -> ConfirmationDecision:
    if len(baseline_by_seed) != 3 or len(candidate_by_seed) != 3:
        raise ValueError("confirmation requires exactly three paired seeds")
    rules = rules or PromotionRules()
    baseline_values, candidate_values = _criterion_values(baseline_by_seed, candidate_by_seed, criterion)
    baseline_mean = statistics.mean(baseline_values)
    candidate_mean = statistics.mean(candidate_values)
    baseline_peak_mean = statistics.mean(summary.peak_accuracy for summary in baseline_by_seed)
    candidate_peak_mean = statistics.mean(summary.peak_accuracy for summary in candidate_by_seed)
    accuracy_constraint_met = candidate_peak_mean >= baseline_peak_mean - rules.maximum_accuracy_loss
    if criterion == "accuracy":
        threshold_met = candidate_mean >= baseline_mean + rules.accuracy_gain
        paired = sum(
            candidate > baseline for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
        )
        differences = [
            candidate - baseline for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
        ]
    elif criterion == "time_to_95":
        threshold_met = candidate_mean <= baseline_mean * (1 - rules.convergence_gain) and accuracy_constraint_met
        paired = sum(
            candidate < baseline for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
        )
        differences = [
            baseline - candidate for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
        ]
    else:
        threshold_met = candidate_mean >= baseline_mean * (1 + rules.auc_gain) and accuracy_constraint_met
        paired = sum(
            candidate > baseline for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
        )
        differences = [
            candidate - baseline for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
        ]
    confirmed = threshold_met and paired >= 2
    reasons = (
        f"three-seed mean {'met' if threshold_met else 'did not meet'} the {criterion} threshold",
        f"{paired} of 3 paired seeds improved in the required direction",
    )
    return ConfirmationDecision(
        confirmed=confirmed,
        criterion=criterion,
        paired_improvements=paired,
        baseline_mean=baseline_mean,
        baseline_std=statistics.stdev(baseline_values),
        candidate_mean=candidate_mean,
        candidate_std=statistics.stdev(candidate_values),
        mean_paired_difference=statistics.mean(differences),
        reasons=reasons,
    )


def _criterion_values(
    baseline: Sequence[ConvergenceSummary],
    candidate: Sequence[ConvergenceSummary],
    criterion: str,
) -> tuple[list[float], list[float]]:
    if criterion == "accuracy":
        return [summary.peak_accuracy for summary in baseline], [summary.peak_accuracy for summary in candidate]
    if criterion == "time_auc":
        return [summary.active_time_auc for summary in baseline], [summary.active_time_auc for summary in candidate]
    baseline_times = [summary.active_seconds_to_95 for summary in baseline]
    candidate_times = [summary.active_seconds_to_95 for summary in candidate]
    if any(value is None for value in (*baseline_times, *candidate_times)):
        raise ValueError("cannot confirm time-to-95 with censored runs")
    return (
        [value for value in baseline_times if value is not None],
        [value for value in candidate_times if value is not None],
    )
