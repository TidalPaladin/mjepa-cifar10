from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, cast

import safetensors.torch as st
import torch
import torch.nn.functional as F
import torchmetrics as tm
import wandb
from mjepa.metrics import CLSPatchAlignmentMetric
from mjepa.model import MJEPA, MJEPAPredictions
from mjepa.optimizer import OptimizerLike, SchedulerLike
from mjepa.trainer import (
    DataLoaderFn,
    TrainerConfig,
    calculate_total_steps,
    format_pbar_description,
    is_rank_zero,
    rank_zero_info,
    save_checkpoint,
    scale_change,
    should_step_optimizer,
    size_change,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torchmetrics.wrappers import Running
from tqdm import tqdm
from vit import ViTFeatures


NUM_CLASSES: Final[int] = 10
WINDOW: Final[int] = 5
LOG_INTERVAL: Final[int] = 50
PERCENT_SCALE: Final[float] = 100.0
GRAD_CLIP_TRIGGER_PCT_KEY: Final[str] = "train/grad_clip_trigger_pct"
CPA_RESULT_KEYS: Final[tuple[str, ...]] = ("cpa_mean", "cpa_std", "cpa_p90", "cpa_p99")


@dataclass(frozen=True)
class OptimizerStepResult:
    next_step: int
    grad_clip_triggered: bool


class CIFAR10MJEPA(MJEPA):
    @staticmethod
    def _flatten_probe_logits(logits: Tensor) -> Tensor:
        if logits.ndim == 2:
            return logits
        if logits.ndim == 3 and logits.shape[1] == 1:
            return logits[:, 0, :]
        raise ValueError(f"probe head must return a single embedding per sample, got shape={tuple(logits.shape)}")

    def forward_probe(self, features: ViTFeatures) -> dict[str, Tensor]:
        probe_tokens = self._get_probe_tokens(features)
        probe_input = probe_tokens.mean(1) if self._has_cls_tokens(features) else probe_tokens
        probe_logits = self.student.get_head("cls")(probe_input)
        return {"cls": self._flatten_probe_logits(probe_logits)}


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


def compute_and_reset_cpa_metrics(metric: CLSPatchAlignmentMetric, prefix: str) -> dict[str, float]:
    cpa_metrics = metric.compute()
    metric.reset()
    return {f"{prefix}/{key}": cpa_metrics[key].item() for key in CPA_RESULT_KEYS}


def update_cls_patch_alignment_metric(metric: CLSPatchAlignmentMetric | None, features: ViTFeatures) -> bool:
    if metric is None or not MJEPA._has_cls_tokens(features):
        return False

    metric.update(features.cls_tokens, features.visual_tokens)
    return True


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


def train(
    jepa: MJEPA | DDP,
    train_dataloader_fn: DataLoaderFn,
    val_dataloader_fn: DataLoaderFn,
    optimizer: OptimizerLike,
    scheduler: SchedulerLike,
    trainer_config: TrainerConfig,
    last_epoch: int = -1,
    max_grad_norm: float | None = None,
) -> None:
    # Module setup
    log_dir = Path(wandb.run.dir) if wandb.run is not None else None
    unwrapped_jepa = jepa.module if isinstance(jepa, DDP) else jepa
    assert isinstance(unwrapped_jepa, MJEPA)
    optimizer.zero_grad()

    # DataLoader setup
    train_dataloader = train_dataloader_fn(unwrapped_jepa.img_size, trainer_config.batch_size)
    val_dataloader = val_dataloader_fn(unwrapped_jepa.img_size, trainer_config.batch_size)
    jepa_scale = unwrapped_jepa.config.scale

    accumulate_grad_batches = trainer_config.accumulate_grad_batches
    microbatch = (last_epoch + 1) * len(train_dataloader)
    step = microbatch // accumulate_grad_batches
    total_steps = calculate_total_steps(train_dataloader, trainer_config.num_epochs, accumulate_grad_batches)
    rank_zero_info(f"Training for {trainer_config.num_epochs} epochs = {total_steps} steps")
    rank_zero_info(
        f"Batch size: {trainer_config.batch_size}, Microbatch accumulation: {trainer_config.accumulate_grad_batches}"
    )

    # Metric setup
    train_loss = tm.RunningMean(window=WINDOW).cuda()
    train_loss_jepa = tm.RunningMean(window=WINDOW).cuda()
    train_loss_jepa_cls = tm.RunningMean(window=WINDOW).cuda()
    train_loss_sigreg = tm.RunningMean(window=WINDOW).cuda()
    train_loss_gram = tm.RunningMean(window=WINDOW).cuda()
    has_jepa_loss_cls = False
    has_sigreg_loss = False
    has_gram_loss = False
    train_acc = Running(tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES), window=WINDOW).cuda()
    train_grad_clip_trigger_pct = tm.MeanMetric().cuda() if max_grad_norm is not None else None
    val_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).cuda()
    train_cpa = CLSPatchAlignmentMetric().cuda() if unwrapped_jepa.student.config.num_cls_tokens > 0 else None
    val_cpa = CLSPatchAlignmentMetric().cuda() if unwrapped_jepa.student.config.num_cls_tokens > 0 else None

    img: Tensor
    label: Tensor
    for epoch in range(last_epoch + 1, trainer_config.num_epochs):
        # Update training resolution / batch_size / accumulate_grad_batches if necessary
        if trainer_config.is_size_change_epoch(epoch):
            size_config = trainer_config.sizes[epoch]
            train_dataloader, val_dataloader, accumulate_grad_batches = size_change(
                size_config,
                trainer_config.batch_size,
                accumulate_grad_batches,
                train_dataloader_fn,
                val_dataloader_fn,
            )
            jepa_scale = scale_change(unwrapped_jepa.img_size, size_config, unwrapped_jepa.config.scale)
            rank_zero_info(
                f"Changing size to {size_config.size} and batch size to {size_config.batch_size} "
                f"(accumulate grad batches: {accumulate_grad_batches}, jepa scale: {jepa_scale})"
            )

        # Update sampler epoch for proper shuffling in DDP
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        jepa.train()
        desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
        pbar = tqdm(train_dataloader, desc=desc, disable=not is_rank_zero(), leave=False)
        for img, label in pbar:
            B = img.shape[0]
            img = img.cuda()
            label = label.cuda()
            should_step = should_step_optimizer(microbatch + 1, accumulate_grad_batches)
            with get_gradient_sync_context(jepa.no_sync if isinstance(jepa, DDP) else None, should_step):
                output = jepa(img, jepa_scale, epoch)
                assert isinstance(output, MJEPAPredictions)
                assert isinstance(unwrapped_jepa, MJEPA)
                ssl_losses = unwrapped_jepa.compute_losses(output, step, epoch)
                train_loss_jepa.update(ssl_losses.jepa_loss)

                jepa_loss_cls = getattr(ssl_losses, "jepa_loss_cls", None)
                if jepa_loss_cls is not None:
                    train_loss_jepa_cls.update(jepa_loss_cls)
                    has_jepa_loss_cls = True

                sigreg_loss = getattr(ssl_losses, "sigreg_loss", None)
                if sigreg_loss is not None:
                    train_loss_sigreg.update(sigreg_loss)
                    has_sigreg_loss = True

                gram_loss = getattr(ssl_losses, "gram_loss", None)
                if gram_loss is not None:
                    train_loss_gram.update(gram_loss)
                    has_gram_loss = True

                ssl_loss = ssl_losses.reduce()

                # Compute linear probe loss
                probe_pred = output.probes["cls"]
                probe_loss = F.cross_entropy(probe_pred, label)

                # Combine losses
                loss = ssl_loss + probe_loss
                train_loss.update(loss)

                with torch.no_grad():
                    train_acc.update(probe_pred, label)
                    update_cls_patch_alignment_metric(train_cpa, output.teacher_output)

                # Backward
                assert not loss.isnan()
                loss.backward()
            unwrapped_jepa.assert_student_params_have_grad(microbatch)
            if isinstance(unwrapped_jepa, MJEPA):
                unwrapped_jepa.assert_predictor_params_have_grad(microbatch)
            microbatch += 1
            should_log_train_metrics = should_step and (step + 1) % LOG_INTERVAL == 0
            grad_norm_stats = None
            if should_log_train_metrics and is_rank_zero():
                grad_norm_stats = get_gradient_norm_stats(unwrapped_jepa.parameters())

            # Optimizer update and teacher update
            if should_step:
                update_teacher = None
                if isinstance(unwrapped_jepa, MJEPA):
                    update_teacher = partial(unwrapped_jepa.update_teacher, step, total_steps)
                optimizer_step_result = run_optimizer_step(
                    optimizer,
                    scheduler,
                    step,
                    total_steps,
                    max_grad_norm=max_grad_norm,
                    update_teacher=update_teacher,
                )
                step = optimizer_step_result.next_step
                if train_grad_clip_trigger_pct is not None:
                    train_grad_clip_trigger_pct.update(float(optimizer_step_result.grad_clip_triggered))

            desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
            pbar.set_description(desc)

            # Log to wandb
            if step % LOG_INTERVAL == 0 and microbatch % accumulate_grad_batches == 0:
                log_dict = {
                    "train/loss": train_loss.compute().item(),
                    "train/loss_jepa": train_loss_jepa.compute().item(),
                    "train/acc": train_acc.compute().item(),
                    "train/lr": get_scheduler_last_lr(scheduler),
                }
                if grad_norm_stats is not None:
                    grad_norm_mean, grad_norm_max = grad_norm_stats
                    log_dict["train/grad_norm_mean"] = grad_norm_mean
                    log_dict["train/grad_norm_max"] = grad_norm_max
                if train_grad_clip_trigger_pct is not None:
                    log_dict[GRAD_CLIP_TRIGGER_PCT_KEY] = compute_and_reset_mean_percentage(train_grad_clip_trigger_pct)
                if has_jepa_loss_cls:
                    log_dict["train/loss_jepa_cls"] = train_loss_jepa_cls.compute().item()
                if has_sigreg_loss:
                    log_dict["train/loss_sigreg"] = train_loss_sigreg.compute().item()
                if has_gram_loss:
                    log_dict["train/loss_gram"] = train_loss_gram.compute().item()
                if train_cpa is not None:
                    log_dict.update(compute_and_reset_cpa_metrics(train_cpa, prefix="train"))
                if is_rank_zero():
                    wandb.log(log_dict, step=step)

        # Validation
        pbar.close()
        unwrapped_jepa.assert_student_params_synced()
        if isinstance(unwrapped_jepa, MJEPA):
            unwrapped_jepa.assert_predictor_params_synced()
        if val_dataloader is not None and (epoch + 1) % trainer_config.check_val_every_n_epoch == 0:
            jepa.eval()
            val_acc.reset()
            if val_cpa is not None:
                val_cpa.reset()

            for img, label in tqdm(val_dataloader, desc="Validating: ", disable=not is_rank_zero(), leave=False):
                B = img.shape[0]
                img = img.cuda()
                label = label.cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = unwrapped_jepa.forward_teacher(img)
                    probe_pred = unwrapped_jepa.forward_probe(output)["cls"].view(B, -1)
                    val_acc.update(probe_pred, label)
                    update_cls_patch_alignment_metric(val_cpa, output)

            # Validation epoch end
            val_acc_value = val_acc.compute()
            rank_zero_info(f"Epoch: {epoch}, Val Acc: {val_acc_value:.4f}")

            # Log validation to wandb
            log_dict = {
                "val/acc": val_acc_value.item(),
                "val/epoch": epoch,
            }
            if val_cpa is not None:
                log_dict.update(compute_and_reset_cpa_metrics(val_cpa, prefix="val"))

            # Add histogram logging
            if is_rank_zero():
                wandb.log(log_dict, step=step)

        # Save checkpoint
        if is_rank_zero() and log_dir:
            save_checkpoint(
                path=log_dir / "checkpoint.pt",
                backbone=unwrapped_jepa.student,
                predictor=unwrapped_jepa.predictor if isinstance(unwrapped_jepa, MJEPA) else None,
                teacher=unwrapped_jepa.teacher if isinstance(unwrapped_jepa, MJEPA) else None,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
            )
            st.save_file(
                {k: v for k, v in unwrapped_jepa.student.state_dict().items() if isinstance(v, torch.Tensor)},
                str(log_dir / "backbone.safetensors"),
            )

    # Save final checkpoint
    if is_rank_zero() and log_dir:
        st.save_file(
            {k: v for k, v in unwrapped_jepa.student.state_dict().items() if isinstance(v, torch.Tensor)},
            str(log_dir / "backbone.safetensors"),
        )
