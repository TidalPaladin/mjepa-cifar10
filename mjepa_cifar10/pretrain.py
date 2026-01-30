import gc
from pathlib import Path
from typing import Final

import safetensors.torch as st
import torch
import torch.nn.functional as F
import torchmetrics as tm
import wandb
from mjepa.model import MJEPA, MJEPAPredictions
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
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DistributedSampler
from torchmetrics.wrappers import Running
from tqdm import tqdm
from vit import ViTFeatures


NUM_CLASSES: Final[int] = 10
WINDOW: Final[int] = 5
LOG_INTERVAL: Final[int] = 50


class CIFAR10MJEPA(MJEPA):

    def forward_probe(self, features: ViTFeatures) -> dict[str, Tensor]:
        return {
            "cls": self.student.get_head("cls")(features.cls_tokens.mean(1)).view(features.cls_tokens.shape[0], -1),
        }


def train(
    jepa: MJEPA | DDP,
    train_dataloader_fn: DataLoaderFn,
    val_dataloader_fn: DataLoaderFn,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    trainer_config: TrainerConfig,
    last_epoch: int = -1,
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
    train_acc = Running(tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES), window=WINDOW).cuda()
    val_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).cuda()

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

            output = jepa(img, jepa_scale, epoch)
            assert isinstance(output, MJEPAPredictions)
            assert isinstance(unwrapped_jepa, MJEPA)
            ssl_losses = unwrapped_jepa.compute_losses(output, step, epoch)
            train_loss_jepa.update(ssl_losses.jepa_loss)
            train_loss_jepa_cls.update(ssl_losses.jepa_loss_cls)
            train_loss_sigreg.update(ssl_losses.sigreg_loss)
            train_loss_gram.update(ssl_losses.gram_loss)
            ssl_loss = ssl_losses.reduce()

            # Compute linear probe loss
            probe_pred = output.probes["cls"]
            probe_loss = F.cross_entropy(probe_pred, label)

            # Combine losses
            loss = ssl_loss + probe_loss
            train_loss.update(loss)

            with torch.no_grad():
                train_acc.update(probe_pred, label)

            # Backward
            assert not loss.isnan()
            loss.backward()
            unwrapped_jepa.assert_student_params_have_grad(microbatch)
            if isinstance(unwrapped_jepa, MJEPA):
                unwrapped_jepa.assert_predictor_params_have_grad(microbatch)
            microbatch += 1

            # Optimizer update and teacher update
            if should_step_optimizer(microbatch, accumulate_grad_batches):
                if step < total_steps:
                    scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                if isinstance(unwrapped_jepa, MJEPA):
                    unwrapped_jepa.update_teacher(step, total_steps)
                step += 1

            desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
            pbar.set_description(desc)

            # Log to wandb
            if step % LOG_INTERVAL == 0 and microbatch % accumulate_grad_batches == 0:
                log_dict = {
                    "train/loss": train_loss.compute().item(),
                    "train/loss_jepa": train_loss_jepa.compute().item(),
                    "train/loss_jepa_cls": train_loss_jepa_cls.compute().item(),
                    "train/loss_sigreg": train_loss_sigreg.compute().item(),
                    "train/loss_gram": train_loss_gram.compute().item(),
                    "train/acc": train_acc.compute().item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }
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

            for img, label in tqdm(val_dataloader, desc=f"Validating: ", disable=not is_rank_zero(), leave=False):
                B = img.shape[0]
                img = img.cuda()
                label = label.cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = unwrapped_jepa.forward_teacher(img)
                    probe_pred = unwrapped_jepa.forward_probe(output)["cls"].view(B, -1)
                    val_acc.update(probe_pred, label)

            # Validation epoch end
            val_acc_value = val_acc.compute()
            rank_zero_info(f"Epoch: {epoch}, Val Acc: {val_acc_value:.4f}")

            # Log validation to wandb
            log_dict = {
                "val/acc": val_acc_value.item(),
                "val/epoch": epoch,
            }

            # Add histogram logging
            if is_rank_zero():
                wandb.log(log_dict, step=step)

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Save checkpoint
        if is_rank_zero() and log_dir:
            save_checkpoint(
                path=log_dir / f"checkpoint.pt",
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
                str(log_dir / f"backbone.safetensors"),
            )

    # Save final checkpoint
    if is_rank_zero() and log_dir:
        st.save_file(
            {k: v for k, v in unwrapped_jepa.student.state_dict().items() if isinstance(v, torch.Tensor)},
            str(log_dir / f"backbone.safetensors"),
        )
