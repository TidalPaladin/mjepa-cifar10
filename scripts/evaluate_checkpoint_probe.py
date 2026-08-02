import hashlib
import json
import os
import tempfile
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Protocol, cast

import torch
import yaml
from mjepa.jepa import CrossAttentionPredictor, JEPAConfig
from mjepa.trainer import TrainerConfig
from torch import Tensor
from torch.utils.data import DataLoader
from vit import ViTConfig

from mjepa_cifar10.data import cifar10_split_fingerprint, get_val_dataloader
from mjepa_cifar10.experiment import append_metric_record
from mjepa_cifar10.pretrain import CIFAR10MJEPA, RAW_MEAN_CLS_GLOBAL_TARGET_POOLING


NUM_CLASSES: Final[int] = 10
HASH_CHUNK_SIZE: Final[int] = 1024 * 1024


class ProbeModel(Protocol):
    training: bool

    def eval(self) -> Any: ...

    def forward_target(self, x: Tensor) -> Any: ...

    def forward_probe(self, features: Any) -> dict[str, Tensor]: ...


@dataclass(frozen=True)
class ProbeEvaluation:
    accuracy: float
    correct: int
    total: int
    elapsed_seconds: float


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate a retained pretraining checkpoint's online probe")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--expected-step", type=int, default=None)
    parser.add_argument("--append-metrics", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_endpoint(checkpoint: Mapping[str, Any]) -> tuple[int, int]:
    step = checkpoint.get("step")
    epoch = checkpoint.get("epoch")
    if isinstance(step, bool) or not isinstance(step, int) or isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError("checkpoint must contain integer step and epoch values")
    return step, epoch


def evaluate_probe(
    model: ProbeModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None,
) -> ProbeEvaluation:
    model.eval()
    correct = 0
    total = 0
    started_at = perf_counter()
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features = model.forward_target(images)
            logits = model.forward_probe(features)["cls"].view(images.shape[0], -1)
            if logits.shape[1] != NUM_CLASSES:
                raise ValueError(f"probe must emit {NUM_CLASSES} logits per image, got {tuple(logits.shape)}")
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.numel()
    if total == 0:
        raise ValueError("validation dataloader produced no examples")
    return ProbeEvaluation(
        accuracy=correct / total,
        correct=correct,
        total=total,
        elapsed_seconds=perf_counter() - started_at,
    )


def load_model(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[CIFAR10MJEPA, TrainerConfig]:
    backbone_config = config.get("backbone")
    jepa_config = config.get("jepa")
    trainer_config = config.get("trainer")
    if not isinstance(backbone_config, ViTConfig):
        raise TypeError("config backbone must be a ViTConfig")
    if not isinstance(jepa_config, JEPAConfig):
        raise TypeError("config jepa must be a JEPAConfig")
    if not isinstance(trainer_config, TrainerConfig):
        raise TypeError("config trainer must be a TrainerConfig")
    checkpoint_image_size = checkpoint.get("img_size")
    if checkpoint_image_size is not None:
        backbone_config = replace(backbone_config, img_size=list(checkpoint_image_size))

    backbone = backbone_config.instantiate(device=device)
    predictor = CrossAttentionPredictor(
        backbone,
        jepa_config.predictor_depth,
        device=device,
        attention_mode=jepa_config.predictor_attention_mode,
        cls_prediction_mode=jepa_config.cls_prediction_mode,
        cls_context_tokens=jepa_config.cls_context_tokens,
        disable_predictor_regularizers=jepa_config.disable_predictor_regularizers,
    )
    pooling = str(config.get("cls_global_target_pooling", RAW_MEAN_CLS_GLOBAL_TARGET_POOLING))
    model = CIFAR10MJEPA(jepa_config, backbone, predictor, cls_global_target_pooling=pooling)
    model.student.load_state_dict(cast(Mapping[str, Tensor], checkpoint["backbone"]))
    model.predictor.load_state_dict(cast(Mapping[str, Tensor], checkpoint["predictor"]))
    teacher_state = checkpoint.get("teacher")
    if model.teacher is None:
        if teacher_state is not None:
            raise ValueError("shared-student config cannot load an EMA teacher state")
    else:
        if not isinstance(teacher_state, Mapping):
            raise ValueError("EMA config requires an EMA teacher state")
        model.teacher.load_state_dict(cast(Mapping[str, Tensor], teacher_state))
    return model, trainer_config


def write_result(path: Path, result: Mapping[str, Any]) -> None:
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
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            temporary_path = Path(output.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def append_endpoint_metric(run_dir: Path, result: Mapping[str, Any]) -> bool:
    checkpoint = cast(Mapping[str, Any], result["checkpoint"])
    evaluation = cast(Mapping[str, Any], result["evaluation"])
    step = int(checkpoint["step"])
    record = {
        "probe/validation_accuracy": float(evaluation["accuracy"]),
        "probe/validation_epoch": int(checkpoint["epoch"]),
        "convergence/active_seconds": float(checkpoint["active_seconds"]),
        "evaluation/terminal_checkpoint_probe": True,
    }
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            existing = json.loads(line)
            if existing.get("_step") == step and existing.get("evaluation/terminal_checkpoint_probe") is True:
                if all(existing.get(key) == value for key, value in record.items()):
                    return False
                raise ValueError(f"conflicting terminal probe metric already exists at step {step}")
    append_metric_record(run_dir, step, record)
    return True


def main(args: Namespace) -> None:
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    config = yaml.full_load(args.config.read_text())
    if not isinstance(config, Mapping):
        raise TypeError("config must contain a mapping")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    step, epoch = checkpoint_endpoint(checkpoint)
    if args.expected_step is not None and step != args.expected_step:
        raise ValueError(f"checkpoint step {step} does not match expected step {args.expected_step}")

    model, trainer_config = load_model(config, checkpoint, device)
    batch_size = args.batch_size or trainer_config.batch_size
    num_workers = trainer_config.num_workers if args.num_workers is None else args.num_workers
    dataloader = get_val_dataloader(
        model.img_size,
        batch_size,
        root=args.data,
        num_workers=num_workers,
    )
    evaluation = evaluate_probe(
        model,
        dataloader,
        device,
        autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    active_seconds = checkpoint.get("elapsed_seconds")
    if not isinstance(active_seconds, (int, float)):
        raise ValueError("checkpoint must contain elapsed_seconds")
    result = {
        "schema_version": 1,
        "kind": "terminal-checkpoint-online-probe",
        "study_id": args.study_id,
        "run_id": args.run_id,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "step": step,
            "epoch": epoch,
            "active_seconds": float(active_seconds),
            "wandb_run_id": checkpoint.get("wandb_run_id"),
        },
        "dataset": {
            "split": "fixed-validation-holdout",
            "split_sha256": cifar10_split_fingerprint(args.data),
            "examples": evaluation.total,
        },
        "evaluation": {
            **asdict(evaluation),
            "model_mode": "eval",
            "gradient_mode": "torch.inference_mode",
            "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
            "device_type": device.type,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "target_encoder": "ema_teacher" if model.teacher is not None else "shared_student",
            "official_test_set_used": False,
        },
    }
    write_result(args.output, result)
    appended = append_endpoint_metric(args.checkpoint.resolve().parent, result) if args.append_metrics else False
    print(json.dumps({**result, "metric_appended": appended}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
