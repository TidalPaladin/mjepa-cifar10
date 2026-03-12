import logging
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Final

import torch
import torch.distributed as dist
import wandb
import yaml
from mjepa.optimizer import OptimizerConfig
from mjepa.trainer import TrainerConfig, calculate_total_steps, ignore_warnings, is_rank_zero, setup_logdir
from torch.nn.parallel import DistributedDataParallel as DDP
from vit import ViTConfig

from mjepa_cifar10.data import get_train_dataloader, get_val_dataloader
from mjepa_cifar10.finetune import CIFAR10FineTuner, load_backbone_checkpoint, train, validate_finetune_config


SEED: Final = 0


def ddp_setup() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        logging.info("Initialized DDP")
    else:
        logging.info("DDP already initialized")


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Cleaned up DDP")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("data", type=Path, help="Path to training data")
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="Name of the run. Will be appended to the log subdirectory."
    )
    parser.add_argument("-l", "--log-dir", type=Path, default=None, help="Directory to save logs")
    parser.add_argument("--local-rank", type=int, default=1, help="Local rank / device")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to backbone safetensors checkpoint")
    return parser.parse_args()


def main(args: Namespace) -> None:
    torch.random.manual_seed(SEED)
    if not (config_path := Path(args.config)).is_file():
        raise FileNotFoundError(config_path)
    config = yaml.full_load(config_path.read_text())
    backbone_config, optimizer_config, trainer_config = validate_finetune_config(config)
    if not isinstance(optimizer_config, OptimizerConfig):
        raise TypeError(f"config['optimizer'] must be an OptimizerConfig, got {type(optimizer_config).__name__}")
    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(trainer_config, TrainerConfig)
    if args.log_dir and not args.log_dir.is_dir():
        raise NotADirectoryError(args.log_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or args.local_rank)
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        ddp_setup()

    run_log_dir = setup_logdir(
        args.log_dir if is_rank_zero() else None,
        config_path if is_rank_zero() else None,
        args.name if is_rank_zero() else None,
    )

    device = torch.device("cuda", local_rank)
    backbone = backbone_config.instantiate(device=device)
    load_backbone_checkpoint(args.checkpoint, backbone, device)
    model = CIFAR10FineTuner(backbone)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    train_dataloader_fn = partial(
        get_train_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
        local_rank=local_rank,
        world_size=world_size,
    )
    val_dataloader_fn = partial(
        get_val_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
    )
    train_dataloader = train_dataloader_fn(unwrapped_model.img_size, trainer_config.batch_size)

    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    optimizer, scheduler = optimizer_config.instantiate(model, total_steps=total_steps)

    if is_rank_zero():
        wandb.init(
            project="mjepa-cifar10",
            name=args.name,
            dir=run_log_dir,
            config={
                "backbone": backbone_config.__dict__,
                "optimizer": optimizer_config.__dict__,
                "trainer": trainer_config.__dict__,
                "checkpoint": str(args.checkpoint),
            },
            tags=("finetune", config_path.stem),
            group="finetune",
        )

    ignore_warnings()
    exit_code = 0
    try:
        train(
            model,
            train_dataloader_fn,
            val_dataloader_fn,
            optimizer,
            scheduler,
            trainer_config,
            max_grad_norm=optimizer_config.max_grad_norm,
        )
    except Exception as error:
        logging.error(f"Error in finetuning: {error}")
        exit_code = 1
        raise
    finally:
        if is_rank_zero():
            wandb.finish(exit_code=exit_code)
        if world_size > 1:
            ddp_cleanup()


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
