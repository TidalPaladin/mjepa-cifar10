from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Self

import yaml


DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ALLOWED_PHYSICAL_GPUS: Final[tuple[int, ...]] = (1, 2)
DEFAULT_MAX_CONCURRENT: Final[int] = 2
DEFAULT_TIMEOUT_SECONDS: Final[int] = 24 * 60 * 60
DEFAULT_MAX_PRETRAIN_TRIALS: Final[int] = 8
DEFAULT_MIN_FREE_GIB: Final[int] = 50
DEFAULT_CHECKPOINT_ESTIMATE_GIB: Final[int] = 3
STUDY_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
RunKind = Literal["pretrain", "sft"]
RunStatus = Literal["pending", "launching", "running", "completed", "failed", "timed_out"]
RunDecision = Literal["pending", "baseline", "promoted", "confirmed", "rejected", "retryable"]


@dataclass(frozen=True)
class VariantSpec:
    id: str
    config: Path
    hypothesis: str
    mechanism: str = ""
    changes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        return cls(
            id=str(value["id"]),
            config=Path(value["config"]),
            hypothesis=str(value.get("hypothesis", "")),
            mechanism=str(value.get("mechanism", "")),
            changes=tuple(str(change) for change in value.get("changes", ())),
        )


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> Self:
        value = value or {}
        return cls(
            accuracy_gain=float(value.get("accuracy_gain", 0.01)),
            convergence_gain=float(value.get("convergence_gain", 0.15)),
            auc_gain=float(value.get("auc_gain", 0.10)),
            maximum_accuracy_loss=float(value.get("maximum_accuracy_loss", 0.005)),
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
class StudySpec:
    id: str
    question: str
    hypothesis: str
    baseline: VariantSpec
    variants: tuple[VariantSpec, ...]
    data: Path
    log_root: Path
    model_class: str = ""
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    wandb_entity: str | None = None
    wandb_project: str = "mjepa-cifar10"
    wandb_group: str = ""
    code_shas: Mapping[str, str] = field(default_factory=dict)
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    promotion: PromotionRules = field(default_factory=PromotionRules)
    evaluation: EvaluationProtocol = field(default_factory=EvaluationProtocol)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, Mapping):
            raise TypeError(f"study specification must be a mapping: {path}")
        spec = cls(
            id=str(raw["id"]),
            question=str(raw["question"]),
            hypothesis=str(raw["hypothesis"]),
            baseline=VariantSpec.from_mapping(raw["baseline"]),
            variants=tuple(VariantSpec.from_mapping(value) for value in raw.get("variants", ())),
            data=Path(os.path.expandvars(str(raw["data"]))),
            log_root=Path(os.path.expandvars(str(raw.get("log_root", "logs/research")))),
            model_class=str(raw.get("model_class", Path(raw["baseline"]["config"]).stem)),
            seeds=tuple(int(seed) for seed in raw.get("seeds", DEFAULT_SEEDS)),
            wandb_entity=raw.get("wandb", {}).get("entity"),
            wandb_project=str(raw.get("wandb", {}).get("project", "mjepa-cifar10")),
            wandb_group=str(raw.get("wandb", {}).get("group", raw["id"])),
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
        if not self.seeds or self.seeds[0] != 0:
            raise ValueError("study seeds must begin with screening seed 0")
        variant_ids = [self.baseline.id, *(variant.id for variant in self.variants)]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("baseline and variant IDs must be unique")
        if require_files:
            root = relative_to or Path.cwd()
            for config in (self.baseline.config, *(variant.config for variant in self.variants)):
                resolved_config = config if config.is_absolute() else root / config
                if not resolved_config.is_file():
                    raise FileNotFoundError(resolved_config)

    def initial_runs(self) -> tuple[RunSpec, ...]:
        screening_variants = (self.baseline, *self.variants[:3])
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
