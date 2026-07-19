"""Persistent, bounded JEPA research orchestration."""

from .metrics import (
    ConvergenceSummary,
    MetricPoint,
    PromotionDecision,
    confirmation_decision,
    derive_convergence_targets,
    promotion_decision,
    summarize_convergence,
)
from .models import RunSpec, RunState, StudySpec, StudyState


__all__ = [
    "ConvergenceSummary",
    "MetricPoint",
    "PromotionDecision",
    "RunSpec",
    "RunState",
    "StudySpec",
    "StudyState",
    "confirmation_decision",
    "derive_convergence_targets",
    "promotion_decision",
    "summarize_convergence",
]
