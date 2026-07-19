from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any, Final, cast

import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
import wandb
from mjepa.optimizer import OptimizerLike, SchedulerLike
from mjepa.trainer import (
    DataLoaderFn,
    TrainerConfig,
    assert_all_ranks_synced,
    assert_all_trainable_params_have_grad,
    calculate_total_steps,
    format_pbar_description,
    is_rank_zero,
    rank_zero_info,
    save_checkpoint,
    should_step_optimizer,
    size_change,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torchmetrics.wrappers import Running
from tqdm import tqdm
from vit import ViT, ViTConfig, ViTFeatures

from .classification import forward_classifier
from .experiment import append_metric_record, save_safetensors_atomic
from .train_utils import (
    compute_and_reset_mean_percentage,
    get_gradient_norm_stats,
    get_gradient_sync_context,
    get_scheduler_last_lr,
    run_optimizer_step,
)


NUM_CLASSES: Final[int] = 10
WINDOW: Final[int] = 5
GRAD_CLIP_TRIGGER_PCT_KEY: Final[str] = "sft/grad_clip_trigger_pct"
SAFETENSORS_SUFFIX: Final[str] = ".safetensors"
REQUIRED_CONFIG_KEYS: Final[tuple[str, ...]] = ("backbone", "optimizer", "trainer")


class CIFAR10FineTuner(nn.Module):
    def __init__(self, backbone: ViT):
        super().__init__()
        self.backbone = backbone

    @property
    def img_size(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.backbone.config.img_size)

    def forward_features(self, img: Tensor) -> ViTFeatures:
        return self.backbone(img)

    def forward_logits(self, features: ViTFeatures) -> Tensor:
        return forward_classifier(self.backbone, features)

    def forward(self, img: Tensor) -> Tensor:
        return self.forward_logits(self.forward_features(img))


def validate_finetune_config(
    config: Mapping[str, Any],
) -> tuple[ViTConfig, Any, TrainerConfig]:
    if "jepa" in config:
        raise ValueError("finetune config must not include a 'jepa' section")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"finetune config is missing required sections: {missing_keys}")

    backbone_config = config["backbone"]
    optimizer_config = config["optimizer"]
    trainer_config = config["trainer"]
    if not isinstance(backbone_config, ViTConfig):
        raise TypeError(f"config['backbone'] must be a ViTConfig, got {type(backbone_config).__name__}")
    if not isinstance(trainer_config, TrainerConfig):
        raise TypeError(f"config['trainer'] must be a TrainerConfig, got {type(trainer_config).__name__}")
    return backbone_config, optimizer_config, trainer_config


def get_autocast_context(device: torch.device) -> AbstractContextManager[None]:
    if device.type != "cuda":
        return nullcontext()
    return cast(AbstractContextManager[None], torch.autocast(device_type=device.type, dtype=torch.bfloat16))


def load_backbone_checkpoint(path: Path, backbone: ViT, device: torch.device) -> None:
    if path.suffix != SAFETENSORS_SUFFIX:
        raise ValueError(f"finetune checkpoint must be a {SAFETENSORS_SUFFIX} file, got {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint_device: str | int = device.type
    if device.type == "cuda" and device.index is not None:
        checkpoint_device = device.index

    state_dict = st.load_file(str(path), device=checkpoint_device)
    backbone.load_state_dict(state_dict, strict=True)


def build_train_log_dict(
    train_loss: tm.Metric,
    train_acc: tm.Metric,
    scheduler: SchedulerLike,
    grad_norm_stats: tuple[float, float] | None = None,
    train_grad_clip_trigger_pct: tm.MeanMetric | None = None,
) -> dict[str, float]:
    log_dict = {
        "sft/train_loss": train_loss.compute().item(),
        "sft/train_accuracy": train_acc.compute().item(),
        "sft/lr": get_scheduler_last_lr(scheduler),
    }
    if grad_norm_stats is not None:
        grad_norm_mean, grad_norm_max = grad_norm_stats
        log_dict["sft/grad_norm_mean"] = grad_norm_mean
        log_dict["sft/grad_norm_max"] = grad_norm_max
    if train_grad_clip_trigger_pct is not None:
        log_dict[GRAD_CLIP_TRIGGER_PCT_KEY] = compute_and_reset_mean_percentage(train_grad_clip_trigger_pct)
    return log_dict


def build_val_log_dict(val_acc: tm.Metric, epoch: int) -> dict[str, float | int]:
    return {
        "sft/validation_accuracy": val_acc.compute().item(),
        "sft/validation_epoch": epoch,
    }


def train(
    model: CIFAR10FineTuner | DDP,
    train_dataloader_fn: DataLoaderFn,
    val_dataloader_fn: DataLoaderFn,
    optimizer: OptimizerLike,
    scheduler: SchedulerLike,
    trainer_config: TrainerConfig,
    test_dataloader_fn: DataLoaderFn | None = None,
    last_epoch: int = -1,
    elapsed_seconds_offset: float = 0.0,
    wandb_run_id: str | None = None,
    output_dir: Path | None = None,
    max_grad_norm: float | None = None,
) -> None:
    training_started_at = perf_counter()
    log_dir = output_dir if output_dir is not None else (Path(wandb.run.dir) if wandb.run is not None else None)
    unwrapped_model = model.module if isinstance(model, DDP) else model
    assert isinstance(unwrapped_model, CIFAR10FineTuner)
    device = next(unwrapped_model.parameters()).device
    optimizer.zero_grad()

    train_dataloader = train_dataloader_fn(unwrapped_model.img_size, trainer_config.batch_size)
    val_dataloader = val_dataloader_fn(unwrapped_model.img_size, trainer_config.batch_size)

    accumulate_grad_batches = trainer_config.accumulate_grad_batches
    microbatch = (last_epoch + 1) * len(train_dataloader)
    step = microbatch // accumulate_grad_batches
    total_steps = calculate_total_steps(train_dataloader, trainer_config.num_epochs, accumulate_grad_batches)
    rank_zero_info(f"Training for {trainer_config.num_epochs} epochs = {total_steps} steps")
    rank_zero_info(
        f"Batch size: {trainer_config.batch_size}, Microbatch accumulation: {trainer_config.accumulate_grad_batches}"
    )

    train_loss = tm.RunningMean(window=WINDOW).to(device)
    train_acc = Running(tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES), window=WINDOW).to(device)
    train_grad_clip_trigger_pct = tm.MeanMetric().to(device) if max_grad_norm is not None else None
    val_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

    img: Tensor
    label: Tensor
    for epoch in range(last_epoch + 1, trainer_config.num_epochs):
        if trainer_config.is_size_change_epoch(epoch):
            size_config = trainer_config.sizes[epoch]
            train_dataloader, val_dataloader, accumulate_grad_batches = size_change(
                size_config,
                trainer_config.batch_size,
                accumulate_grad_batches,
                train_dataloader_fn,
                val_dataloader_fn,
            )
            rank_zero_info(
                f"Changing size to {size_config.size} and batch size to {size_config.batch_size} "
                f"(accumulate grad batches: {accumulate_grad_batches})"
            )

        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
        pbar = tqdm(train_dataloader, desc=desc, disable=not is_rank_zero(), leave=False)
        for img, label in pbar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            should_step = should_step_optimizer(microbatch + 1, accumulate_grad_batches)

            with get_gradient_sync_context(model.no_sync if isinstance(model, DDP) else None, should_step):
                with get_autocast_context(img.device):
                    logits = model(img)
                    loss = F.cross_entropy(logits, label)

                train_loss.update(loss.detach())
                with torch.no_grad():
                    train_acc.update(logits.detach(), label)

                assert not loss.isnan()
                loss.backward()

            assert_all_trainable_params_have_grad(unwrapped_model, microbatch)
            microbatch += 1

            should_log_train_metrics = should_step and (step + 1) % trainer_config.log_interval == 0
            grad_norm_stats = None
            if should_log_train_metrics and is_rank_zero():
                grad_norm_stats = get_gradient_norm_stats(unwrapped_model.parameters())

            if should_step:
                optimizer_step_result = run_optimizer_step(
                    optimizer,
                    scheduler,
                    step,
                    total_steps,
                    max_grad_norm=max_grad_norm,
                )
                step = optimizer_step_result.next_step
                if train_grad_clip_trigger_pct is not None:
                    train_grad_clip_trigger_pct.update(float(optimizer_step_result.grad_clip_triggered))

            desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
            pbar.set_description(desc)

            if step % trainer_config.log_interval == 0 and microbatch % accumulate_grad_batches == 0:
                log_dict = build_train_log_dict(
                    train_loss,
                    train_acc,
                    scheduler,
                    grad_norm_stats=grad_norm_stats,
                    train_grad_clip_trigger_pct=train_grad_clip_trigger_pct,
                )
                log_dict["convergence/active_seconds"] = elapsed_seconds_offset + perf_counter() - training_started_at
                if is_rank_zero():
                    wandb.log(log_dict, step=step)

        pbar.close()
        assert_all_ranks_synced(unwrapped_model)
        if val_dataloader is not None and (epoch + 1) % trainer_config.check_val_every_n_epoch == 0:
            model.eval()
            val_acc.reset()

            for img, label in tqdm(val_dataloader, desc="Validating: ", disable=not is_rank_zero(), leave=False):
                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                with torch.inference_mode(), get_autocast_context(img.device):
                    logits = unwrapped_model(img)
                    val_acc.update(logits, label)

            val_acc_value = val_acc.compute()
            rank_zero_info(f"Epoch: {epoch}, Val Acc: {val_acc_value:.4f}")

            if is_rank_zero():
                val_log_dict = build_val_log_dict(val_acc, epoch)
                val_log_dict["convergence/active_seconds"] = (
                    elapsed_seconds_offset + perf_counter() - training_started_at
                )
                wandb.log(val_log_dict, step=step)
                append_metric_record(log_dir, step, val_log_dict)

        if is_rank_zero() and log_dir:
            save_checkpoint(
                path=log_dir / "checkpoint.pt",
                backbone=unwrapped_model.backbone,
                predictor=None,
                teacher=None,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
                elapsed_seconds=elapsed_seconds_offset + perf_counter() - training_started_at,
                wandb_run_id=wandb_run_id,
            )
            save_safetensors_atomic(
                log_dir / "backbone.safetensors",
                {k: v for k, v in unwrapped_model.backbone.state_dict().items() if isinstance(v, Tensor)},
            )

    if is_rank_zero() and log_dir:
        save_safetensors_atomic(
            log_dir / "backbone.safetensors",
            {k: v for k, v in unwrapped_model.backbone.state_dict().items() if isinstance(v, Tensor)},
        )

    if test_dataloader_fn is not None:
        test_dataloader = test_dataloader_fn(unwrapped_model.img_size, trainer_config.batch_size)
        test_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
        model.eval()
        for img, label in tqdm(test_dataloader, desc="Testing: ", disable=not is_rank_zero(), leave=False):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.inference_mode(), get_autocast_context(img.device):
                test_acc.update(unwrapped_model(img), label)
        if is_rank_zero():
            test_log_dict = {
                "sft/test_accuracy": test_acc.compute().item(),
                "convergence/active_seconds": elapsed_seconds_offset + perf_counter() - training_started_at,
            }
            wandb.log(test_log_dict, step=step)
            append_metric_record(log_dir, step, test_log_dict)
