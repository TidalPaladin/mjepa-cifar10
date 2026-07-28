from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Final, cast

import torch
from mjepa.jepa import (
    JOINT_CONTEXT_CLS_PREDICTION_MODES,
    PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES,
    CrossAttentionPredictor,
    joint_context_query_multiplier,
)
from mjepa.model import MJEPA


DEFAULT_BATCH_SIZE: Final[int] = 512
DEFAULT_TARGET_QUERIES: Final[int] = 16
DEFAULT_WARMUP_ITERATIONS: Final[int] = 20
DEFAULT_MEASURED_ITERATIONS: Final[int] = 100


@dataclass(frozen=True)
class CLSPathBenchmarkResult:
    cls_prediction_mode: str
    benchmark_scope: str
    median_ms: float
    p90_ms: float
    parameter_count: int
    batch_size: int
    visual_context_tokens: int
    target_queries: int
    executed_target_queries: int
    predictor_forward_count: int
    warmup_iterations: int
    measured_iterations: int
    autocast_dtype: str
    gpu_name: str
    gpu_total_memory_bytes: int
    gpu_compute_capability: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_metrics(self) -> dict[str, float | int]:
        return {
            "diagnostics/cls_path_latency_median_ms": self.median_ms,
            "diagnostics/cls_path_latency_p90_ms": self.p90_ms,
            "diagnostics/cls_path_parameter_count": self.parameter_count,
        }


def _unique_parameter_count(modules: tuple[torch.nn.Module, ...]) -> int:
    parameters = {id(parameter): parameter for module in modules for parameter in module.parameters()}
    return sum(parameter.numel() for parameter in parameters.values())


def count_cls_prediction_path_parameters(predictor: CrossAttentionPredictor) -> int:
    """Count unique predictor parameters participating in the complete predictor workload."""
    return _unique_parameter_count((predictor,))


def _run_cls_prediction_path(
    jepa: MJEPA,
    tokenized_size: tuple[int, int],
    visual_context: torch.Tensor,
    cls_tokens: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute the predictor work performed by one training step."""
    if jepa.config.cls_prediction_mode in JOINT_CONTEXT_CLS_PREDICTION_MODES:
        return jepa.forward_joint_context_predictor_heads(
            tokenized_size,
            visual_context,
            cls_tokens,
            context_mask,
            target_mask,
            rope_seed=0,
        )
    if jepa.config.cls_prediction_mode in PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES:
        return jepa.forward_packed_adaln_hard_blind_predictor_heads(
            tokenized_size,
            visual_context,
            cls_tokens,
            context_mask,
            target_mask,
            rope_seed=0,
        )
    visual_prediction = jepa.forward_predictor(
        tokenized_size,
        visual_context,
        context_mask,
        target_mask,
        rope_seed=0,
    )
    cls_prediction = jepa.forward_cls_predictor(tokenized_size, cls_tokens, target_mask, rope_seed=0)
    return visual_prediction, cls_prediction


def benchmark_cls_prediction_path(
    jepa: MJEPA,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_queries: int = DEFAULT_TARGET_QUERIES,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    measured_iterations: int = DEFAULT_MEASURED_ITERATIONS,
) -> CLSPathBenchmarkResult:
    """Measure the complete configured predictor workload with CUDA events."""
    device = next(jepa.predictor.parameters()).device
    if device.type != "cuda":
        raise ValueError("CLS path benchmark requires a CUDA model")
    raw_tokenized_size = jepa.student.stem.tokenized_size(jepa.img_size)
    if len(raw_tokenized_size) != 2:
        raise ValueError("CLS path benchmark requires a two-dimensional token grid")
    tokenized_size = cast(tuple[int, int], raw_tokenized_size)
    visual_tokens = math.prod(tokenized_size)
    if not 0 < target_queries <= visual_tokens:
        raise ValueError(f"target_queries must be between 1 and {visual_tokens}")
    if batch_size <= 0 or warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("benchmark batch and iteration counts must be positive")

    target_mask = torch.zeros(batch_size, visual_tokens, dtype=torch.bool, device=device)
    target_mask[:, :target_queries] = True
    visual_context_tokens = max(1, min(round(visual_tokens * jepa.config.context_ratio), visual_tokens))
    context_mask = torch.zeros(batch_size, visual_tokens, dtype=torch.bool, device=device)
    context_mask[:, :visual_context_tokens] = True
    hidden_size = jepa.predictor.hidden_size
    visual_context = torch.zeros(
        batch_size,
        visual_context_tokens,
        hidden_size,
        dtype=jepa.predictor.predictor_dtype,
        device=device,
    )
    cls_token_count = jepa.student.config.num_cls_tokens
    cls_tokens = torch.zeros(
        batch_size,
        cls_token_count,
        hidden_size,
        dtype=jepa.predictor.predictor_dtype,
        device=device,
    )

    def run_path() -> None:
        _run_cls_prediction_path(
            jepa,
            tokenized_size,
            visual_context,
            cls_tokens,
            context_mask,
            target_mask,
        )

    was_training = jepa.predictor.training
    jepa.predictor.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(warmup_iterations):
                run_path()
            torch.cuda.synchronize(device)
            elapsed_ms: list[float] = []
            for _ in range(measured_iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_path()
                end.record()
                end.synchronize()
                elapsed_ms.append(float(start.elapsed_time(end)))
    finally:
        jepa.predictor.train(was_training)

    ordered = sorted(elapsed_ms)
    p90_index = max(math.ceil(0.9 * len(ordered)) - 1, 0)
    properties = torch.cuda.get_device_properties(device)
    return CLSPathBenchmarkResult(
        cls_prediction_mode=jepa.config.cls_prediction_mode,
        benchmark_scope="complete_predictor_workload",
        median_ms=median(ordered),
        p90_ms=ordered[p90_index],
        parameter_count=count_cls_prediction_path_parameters(jepa.predictor),
        batch_size=batch_size,
        visual_context_tokens=visual_context_tokens,
        target_queries=target_queries,
        executed_target_queries=target_queries * joint_context_query_multiplier(jepa.config.cls_prediction_mode),
        predictor_forward_count=(
            1
            if jepa.config.cls_prediction_mode
            in (*JOINT_CONTEXT_CLS_PREDICTION_MODES, *PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES)
            else 2
        ),
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        autocast_dtype="bfloat16",
        gpu_name=properties.name,
        gpu_total_memory_bytes=properties.total_memory,
        gpu_compute_capability=f"{properties.major}.{properties.minor}",
    )


def write_cls_path_benchmark(path: Path, result: CLSPathBenchmarkResult) -> None:
    """Atomically persist benchmark inputs, hardware identity, and results."""
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
            json.dump(result.to_dict(), output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
