from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

import torch
import torchmetrics as tm
from mjepa.optimizer import OptimizerLike, SchedulerLike
from torch import Tensor


PERCENT_SCALE = 100.0


@dataclass(frozen=True)
class OptimizerStepResult:
    next_step: int
    grad_clip_triggered: bool


def get_scheduler_last_lr(scheduler: SchedulerLike) -> float:
    get_last_lr = getattr(scheduler, "get_last_lr", None)
    if not callable(get_last_lr):
        raise TypeError("scheduler must expose get_last_lr() for training metrics")

    last_lr = cast(list[float], get_last_lr())
    if not last_lr:
        raise ValueError("scheduler.get_last_lr() returned no learning rates")
    return float(last_lr[0])


def get_gradient_norm_stats(parameters: Iterable[torch.nn.Parameter]) -> tuple[float, float] | None:
    grad_norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not grad_norms:
        return None

    grad_norm_tensor = torch.stack(grad_norms)
    return grad_norm_tensor.mean().item(), grad_norm_tensor.max().item()


def get_gradient_sync_context(
    no_sync: Callable[[], AbstractContextManager[None]] | None,
    should_sync_gradients: bool,
) -> AbstractContextManager[None]:
    if should_sync_gradients or no_sync is None:
        return nullcontext()
    return no_sync()


def clip_optimizer_grad_norm_(optimizer: OptimizerLike, max_grad_norm: float | None) -> Tensor | None:
    if max_grad_norm is None:
        return None

    seen_parameter_ids: set[int] = set()
    unique_parameters: list[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        for parameter in cast(Iterable[torch.nn.Parameter], group["params"]):
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                continue
            seen_parameter_ids.add(parameter_id)
            unique_parameters.append(parameter)

    return torch.nn.utils.clip_grad_norm_(unique_parameters, max_norm=max_grad_norm)


def did_gradient_clip(total_grad_norm: Tensor | None, max_grad_norm: float | None) -> bool:
    if total_grad_norm is None or max_grad_norm is None:
        return False
    return bool(total_grad_norm.item() > max_grad_norm)


def compute_and_reset_mean_percentage(metric: tm.MeanMetric) -> float:
    percentage = metric.compute().item() * PERCENT_SCALE
    metric.reset()
    return percentage


def run_optimizer_step(
    optimizer: OptimizerLike,
    scheduler: SchedulerLike,
    step: int,
    total_steps: int,
    max_grad_norm: float | None = None,
    update_teacher: Callable[[], None] | None = None,
) -> OptimizerStepResult:
    total_grad_norm = clip_optimizer_grad_norm_(optimizer, max_grad_norm)
    if step < total_steps:
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()
    if update_teacher is not None:
        update_teacher()
    return OptimizerStepResult(
        next_step=step + 1,
        grad_clip_triggered=did_gradient_clip(total_grad_norm, max_grad_norm),
    )
