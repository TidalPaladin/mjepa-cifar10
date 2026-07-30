from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, Self

import yaml


DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ALLOWED_PHYSICAL_GPUS: Final[tuple[int, ...]] = (1, 2)
DEFAULT_MAX_CONCURRENT: Final[int] = 2
DEFAULT_TIMEOUT_SECONDS: Final[int] = 24 * 60 * 60
DEFAULT_MAX_PRETRAIN_TRIALS: Final[int] = 8
DEFAULT_MIN_FREE_GIB: Final[int] = 50
DEFAULT_CHECKPOINT_ESTIMATE_GIB: Final[int] = 3
WANDB_OPERATION_EMITTED_DATA_CLASSES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "launch": frozenset(("metrics", "configs", "provenance")),
        "summary": frozenset(("metrics", "provenance")),
    }
)
WANDB_LOCAL_MODES: Final[frozenset[str]] = frozenset(("offline", "disabled", "dryrun"))
STUDY_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
RunKind = Literal["pretrain", "sft"]
WandbOperation = Literal["launch", "summary"]
WandbEffectiveMode = Literal["online", "local-only"]
RunStatus = Literal["pending", "launching", "running", "completed", "failed", "timed_out"]
RunDecision = Literal["pending", "baseline", "promoted", "confirmed", "rejected", "retryable"]


@dataclass(frozen=True)
class VariantSpec:
    id: str
    config: Path
    hypothesis: str
    mechanism: str = ""
    changes: tuple[str, ...] = ()
    finetune_config: Path | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        return cls(
            id=str(value["id"]),
            config=Path(value["config"]),
            hypothesis=str(value.get("hypothesis", "")),
            mechanism=str(value.get("mechanism", "")),
            changes=tuple(str(change) for change in value.get("changes", ())),
            finetune_config=Path(value["finetune_config"]) if value.get("finetune_config") else None,
        )


@dataclass(frozen=True)
class BaselineReference:
    study_id: str
    run_id: str
    metrics: Path
    metrics_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        return cls(
            study_id=str(value["study_id"]),
            run_id=str(value["run_id"]),
            metrics=Path(os.path.expandvars(str(value["metrics"]))),
            metrics_sha256=str(value["metrics_sha256"]),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "study_id": self.study_id,
            "run_id": self.run_id,
            "metrics": str(self.metrics),
            "metrics_sha256": self.metrics_sha256,
        }


@dataclass(frozen=True)
class ResourceLimits:
    physical_gpus: tuple[int, ...] = ALLOWED_PHYSICAL_GPUS
    max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_pretraining_trials: int = DEFAULT_MAX_PRETRAIN_TRIALS
    minimum_free_gib: int = DEFAULT_MIN_FREE_GIB
    fallback_checkpoint_gib: int = DEFAULT_CHECKPOINT_ESTIMATE_GIB

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> Self:
        value = value or {}
        return cls(
            physical_gpus=tuple(int(gpu) for gpu in value.get("physical_gpus", ALLOWED_PHYSICAL_GPUS)),
            max_concurrent_jobs=int(value.get("max_concurrent_jobs", DEFAULT_MAX_CONCURRENT)),
            timeout_seconds=int(value.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
            max_pretraining_trials=int(value.get("max_pretraining_trials", DEFAULT_MAX_PRETRAIN_TRIALS)),
            minimum_free_gib=int(value.get("minimum_free_gib", DEFAULT_MIN_FREE_GIB)),
            fallback_checkpoint_gib=int(value.get("fallback_checkpoint_gib", DEFAULT_CHECKPOINT_ESTIMATE_GIB)),
        )


@dataclass(frozen=True)
class PromotionRules:
    accuracy_gain: float = 0.01
    convergence_gain: float = 0.15
    auc_gain: float = 0.10
    maximum_accuracy_loss: float = 0.005
    cost_gain: float | None = None
    equivalence_convergence_ratio: float | None = None
    equivalence_auc_loss: float | None = None
    screening_control_variant: str | None = None
    screening_control_accuracy_gain: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> Self:
        value = value or {}
        return cls(
            accuracy_gain=float(value.get("accuracy_gain", 0.01)),
            convergence_gain=float(value.get("convergence_gain", 0.15)),
            auc_gain=float(value.get("auc_gain", 0.10)),
            maximum_accuracy_loss=float(value.get("maximum_accuracy_loss", 0.005)),
            cost_gain=float(value["cost_gain"]) if value.get("cost_gain") is not None else None,
            equivalence_convergence_ratio=(
                float(value["equivalence_convergence_ratio"])
                if value.get("equivalence_convergence_ratio") is not None
                else None
            ),
            equivalence_auc_loss=(
                float(value["equivalence_auc_loss"]) if value.get("equivalence_auc_loss") is not None else None
            ),
            screening_control_variant=(
                str(value["screening_control_variant"]) if value.get("screening_control_variant") is not None else None
            ),
            screening_control_accuracy_gain=(
                float(value["screening_control_accuracy_gain"])
                if value.get("screening_control_accuracy_gain") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class EvaluationProtocol:
    finetune_config: Path | None = None
    shots_per_class: tuple[int | None, ...] = (None, 10, 100)
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    official_test_roles: tuple[str, ...] = ("baseline", "winner")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> Self:
        value = value or {}
        finetune_config = value.get("finetune_config")
        raw_shots = value.get("shots_per_class", [None, 10, 100])
        return cls(
            finetune_config=Path(finetune_config) if finetune_config else None,
            shots_per_class=tuple(None if shot is None else int(shot) for shot in raw_shots),
            seeds=tuple(int(seed) for seed in value.get("seeds", DEFAULT_SEEDS)),
            official_test_roles=tuple(str(role) for role in value.get("official_test_roles", ("baseline", "winner"))),
        )


@dataclass(frozen=True)
class WandbOperationDecision:
    operation: WandbOperation
    requested_mode: str
    effective_mode: WandbEffectiveMode
    destination: str | None
    emitted_data_classes: tuple[str, ...]
    approved_data_classes: tuple[str, ...]
    missing_data_classes: tuple[str, ...]
    authorized: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudySpec:
    id: str
    question: str
    hypothesis: str
    baseline: VariantSpec
    variants: tuple[VariantSpec, ...]
    data: Path
    log_root: Path
    baseline_reference: BaselineReference | None = None
    model_class: str = ""
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    wandb_entity: str | None = None
    wandb_project: str = "mjepa-cifar10"
    wandb_group: str = ""
    wandb_authorized: bool = False
    wandb_approved_data_classes: tuple[str, ...] = ()
    wandb_emitted_data_classes: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            operation: tuple(sorted(classes)) for operation, classes in WANDB_OPERATION_EMITTED_DATA_CLASSES.items()
        }
    )
    wandb_manifests_explicit: bool = True
    code_shas: Mapping[str, str] = field(default_factory=dict)
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    promotion: PromotionRules = field(default_factory=PromotionRules)
    evaluation: EvaluationProtocol = field(default_factory=EvaluationProtocol)

    @property
    def wandb_online_authorized(self) -> bool:
        return self.wandb_operation_decision("launch", "online").authorized

    def finetune_config_for(self, variant_id: str) -> Path | None:
        variant_by_id = {variant.id: variant for variant in (self.baseline, *self.variants)}
        try:
            variant = variant_by_id[variant_id]
        except KeyError:
            raise ValueError(f"unknown study variant {variant_id!r}") from None
        return variant.finetune_config or self.evaluation.finetune_config

    def wandb_operation_decision(self, operation: WandbOperation, requested_mode: str) -> WandbOperationDecision:
        emitted = tuple(sorted(self.wandb_emitted_data_classes[operation]))
        approved = tuple(sorted(set(self.wandb_approved_data_classes)))
        missing = tuple(sorted(set(emitted) - set(approved)))
        normalized_mode = requested_mode.strip().lower() or "online"
        requested_local = normalized_mode in WANDB_LOCAL_MODES
        destination = f"{self.wandb_entity}/{self.wandb_project}" if self.wandb_entity else None
        authorized = bool(
            not requested_local
            and destination
            and self.wandb_authorized
            and self.wandb_manifests_explicit
            and not missing
        )
        reasons: list[str] = []
        if requested_local:
            reasons.append("local mode requested")
        if not destination:
            reasons.append("destination entity is missing")
        if not self.wandb_authorized:
            reasons.append("external publication is not authorized")
        if not self.wandb_manifests_explicit:
            reasons.append("the emitted-data manifest is not explicit")
        if missing:
            reasons.append(f"approval is missing for: {', '.join(missing)}")
        return WandbOperationDecision(
            operation=operation,
            requested_mode=normalized_mode,
            effective_mode="online" if authorized else "local-only",
            destination=destination,
            emitted_data_classes=emitted,
            approved_data_classes=approved,
            missing_data_classes=missing,
            authorized=authorized,
            reason="online operation authorized" if authorized else "; ".join(reasons),
        )

    @classmethod
    def from_path(cls, path: Path) -> Self:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, Mapping):
            raise TypeError(f"study specification must be a mapping: {path}")
        wandb = raw.get("wandb", {})
        if not isinstance(wandb, Mapping):
            raise TypeError(f"wandb study configuration must be a mapping: {path}")
        manifests_explicit = "emitted_data_classes" in wandb
        raw_manifests = wandb.get("emitted_data_classes", WANDB_OPERATION_EMITTED_DATA_CLASSES)
        if not isinstance(raw_manifests, Mapping):
            raise TypeError(f"wandb emitted_data_classes must be a mapping: {path}")
        manifests = {
            str(operation): tuple(sorted(str(value) for value in values)) for operation, values in raw_manifests.items()
        }
        spec = cls(
            id=str(raw["id"]),
            question=str(raw["question"]),
            hypothesis=str(raw["hypothesis"]),
            baseline=VariantSpec.from_mapping(raw["baseline"]),
            variants=tuple(VariantSpec.from_mapping(value) for value in raw.get("variants", ())),
            data=Path(os.path.expandvars(str(raw["data"]))),
            log_root=Path(os.path.expandvars(str(raw.get("log_root", "logs/research")))),
            baseline_reference=(
                BaselineReference.from_mapping(raw["baseline_reference"])
                if raw.get("baseline_reference") is not None
                else None
            ),
            model_class=str(raw.get("model_class", Path(raw["baseline"]["config"]).stem)),
            seeds=tuple(int(seed) for seed in raw.get("seeds", DEFAULT_SEEDS)),
            wandb_entity=wandb.get("entity"),
            wandb_project=str(wandb.get("project", "mjepa-cifar10")),
            wandb_group=str(wandb.get("group", raw["id"])),
            wandb_authorized=bool(wandb.get("authorized", False)),
            wandb_approved_data_classes=tuple(str(value) for value in wandb.get("approved_data_classes", ())),
            wandb_emitted_data_classes=manifests,
            wandb_manifests_explicit=manifests_explicit,
            code_shas={str(key): str(value) for key, value in raw.get("code_shas", {}).items()},
            resources=ResourceLimits.from_mapping(raw.get("resources")),
            promotion=PromotionRules.from_mapping(raw.get("promotion")),
            evaluation=EvaluationProtocol.from_mapping(raw.get("evaluation")),
        )
        spec.validate(require_files=False)
        return spec

    def validate(self, relative_to: Path | None = None, *, require_files: bool = True) -> None:
        if not STUDY_ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"invalid study ID {self.id!r}; use lowercase letters, digits, and hyphens")
        if self.resources.physical_gpus != ALLOWED_PHYSICAL_GPUS:
            raise ValueError(f"managed studies must use physical GPUs {ALLOWED_PHYSICAL_GPUS}")
        if self.resources.max_concurrent_jobs > len(ALLOWED_PHYSICAL_GPUS):
            raise ValueError("max_concurrent_jobs exceeds the managed GPU count")
        if self.resources.max_pretraining_trials > DEFAULT_MAX_PRETRAIN_TRIALS:
            raise ValueError(f"pretraining trial limit cannot exceed {DEFAULT_MAX_PRETRAIN_TRIALS}")
        if self.promotion.cost_gain is not None and not 0 < self.promotion.cost_gain < 1:
            raise ValueError("promotion cost_gain must be between 0 and 1")
        if (self.promotion.equivalence_convergence_ratio is None) != (self.promotion.equivalence_auc_loss is None):
            raise ValueError(
                "promotion equivalence_convergence_ratio and equivalence_auc_loss must be configured together"
            )
        if (
            self.promotion.equivalence_convergence_ratio is not None
            and self.promotion.equivalence_convergence_ratio < 1
        ):
            raise ValueError("promotion equivalence_convergence_ratio must be at least 1")
        if self.promotion.equivalence_auc_loss is not None and self.promotion.equivalence_auc_loss < 0:
            raise ValueError("promotion equivalence_auc_loss must be non-negative")
        if not self.seeds or self.seeds[0] != 0:
            raise ValueError("study seeds must begin with screening seed 0")
        expected_manifests = {
            operation: tuple(sorted(classes)) for operation, classes in WANDB_OPERATION_EMITTED_DATA_CLASSES.items()
        }
        if dict(self.wandb_emitted_data_classes) != expected_manifests:
            raise ValueError(f"W&B emitted-data manifests must match the adapter: {expected_manifests}")
        variant_ids = [self.baseline.id, *(variant.id for variant in self.variants)]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("baseline and variant IDs must be unique")
        screening_control_variant = self.promotion.screening_control_variant
        screening_control_accuracy_gain = self.promotion.screening_control_accuracy_gain
        if (screening_control_variant is None) != (screening_control_accuracy_gain is None):
            raise ValueError("screening control variant and accuracy gain must be configured together")
        if screening_control_variant is not None:
            candidate_ids = {variant.id for variant in self.variants}
            if screening_control_variant not in candidate_ids:
                raise ValueError("screening control variant must name a configured non-baseline variant")
            assert screening_control_accuracy_gain is not None
            if screening_control_accuracy_gain <= 0:
                raise ValueError("screening control accuracy gain must be positive")
        if self.baseline_reference is not None:
            if len(self.variants) > self.resources.max_pretraining_trials:
                raise ValueError("reference-baseline candidate count exceeds the pretraining trial limit")
            if not re.fullmatch(r"[0-9a-f]{64}", self.baseline_reference.metrics_sha256):
                raise ValueError("baseline reference metrics_sha256 must be a lowercase SHA-256 digest")
        if require_files:
            root = relative_to or Path.cwd()
            for config in (self.baseline.config, *(variant.config for variant in self.variants)):
                resolved_config = config if config.is_absolute() else root / config
                if not resolved_config.is_file():
                    raise FileNotFoundError(resolved_config)
            finetune_configs = {
                config
                for config in (
                    self.evaluation.finetune_config,
                    self.baseline.finetune_config,
                    *(variant.finetune_config for variant in self.variants),
                )
                if config is not None
            }
            for config in finetune_configs:
                resolved_config = config if config.is_absolute() else root / config
                if not resolved_config.is_file():
                    raise FileNotFoundError(resolved_config)
            if self.baseline_reference is not None:
                metrics = self.baseline_reference.metrics
                resolved_metrics = metrics if metrics.is_absolute() else root / metrics
                if not resolved_metrics.is_file():
                    raise FileNotFoundError(resolved_metrics)
                observed_sha256 = hashlib.sha256(resolved_metrics.read_bytes()).hexdigest()
                if observed_sha256 != self.baseline_reference.metrics_sha256:
                    raise ValueError(
                        "baseline reference metrics hash mismatch: "
                        f"expected {self.baseline_reference.metrics_sha256}, observed {observed_sha256}"
                    )

    def initial_runs(self) -> tuple[RunSpec, ...]:
        screening_variants = (
            self.variants if self.baseline_reference is not None else (self.baseline, *self.variants[:3])
        )
        return tuple(
            RunSpec(
                id=f"pretrain-{variant.id}-seed0",
                kind="pretrain",
                variant=variant.id,
                config=variant.config,
                seed=0,
                role="baseline" if variant.id == self.baseline.id else "candidate",
                evaluate_test=False,
            )
            for variant in screening_variants
        )


@dataclass(frozen=True)
class RunSpec:
    id: str
    kind: RunKind
    variant: str
    config: Path
    seed: int
    role: str
    source_checkpoint: Path | None = None
    shots_per_class: int | None = None
    subset_seed: int | None = None
    evaluate_test: bool = False
    command: tuple[str, ...] | None = None


@dataclass
class RunState:
    spec: RunSpec
    status: RunStatus = "pending"
    decision: RunDecision = "pending"
    physical_gpu: int | None = None
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    wandb_run_id: str | None = None
    wandb_url: str | None = None
    run_dir: str | None = None
    checkpoint_disposition: str = "retained"
    bytes_freed: int = 0
    attempt: int = 1
    originating_thread_id: str | None = None
    heartbeat_at: str | None = None
    current_progress: float | None = None
    routine_check_count: int = 0
    last_check_at: str | None = None
    last_check_interval_seconds: float | None = None
    next_check_at: str | None = None
    next_check_reason: str | None = None
    terminal_event_id: str | None = None
    notification_attempts: int = 0
    notification_last_error: str | None = None
    notification_next_attempt_at: str | None = None
    notification_accepted_at: str | None = None
    notification_accepted_rpc_method: str | None = None
    notification_accepted_turn_id: str | None = None
    notification_state: str = "not-requested"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["spec"]["config"] = str(self.spec.config)
        value["spec"]["source_checkpoint"] = (
            str(self.spec.source_checkpoint) if self.spec.source_checkpoint is not None else None
        )
        value["spec"]["command"] = list(self.spec.command) if self.spec.command is not None else None
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        raw_spec = value["spec"]
        spec = RunSpec(
            id=raw_spec["id"],
            kind=raw_spec["kind"],
            variant=raw_spec["variant"],
            config=Path(raw_spec["config"]),
            seed=int(raw_spec["seed"]),
            role=raw_spec["role"],
            source_checkpoint=Path(raw_spec["source_checkpoint"]) if raw_spec.get("source_checkpoint") else None,
            shots_per_class=raw_spec.get("shots_per_class"),
            subset_seed=raw_spec.get("subset_seed"),
            evaluate_test=bool(raw_spec.get("evaluate_test", False)),
            command=tuple(raw_spec["command"]) if raw_spec.get("command") else None,
        )
        state_values = {key: field_value for key, field_value in value.items() if key != "spec"}
        return cls(spec=spec, **state_values)


@dataclass
class StudyState:
    study_id: str
    spec_path: str
    created_at: str
    updated_at: str
    runs: dict[str, RunState]
    phase: str = "screening"
    winner: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "spec_path": self.spec_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phase": self.phase,
            "winner": self.winner,
            "runs": {run_id: run.to_dict() for run_id, run in self.runs.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            study_id=str(value["study_id"]),
            spec_path=str(value["spec_path"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            phase=str(value.get("phase", "screening")),
            winner=value.get("winner"),
            runs={run_id: RunState.from_dict(run) for run_id, run in value["runs"].items()},
        )
