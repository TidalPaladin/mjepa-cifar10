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
from mjepa.jepa import CrossAttentionPredictor, JEPAConfig
from mjepa.optimizer import OptimizerConfig
from mjepa.trainer import TrainerConfig, calculate_total_steps, ignore_warnings, is_rank_zero
from torch.nn.parallel import DistributedDataParallel as DDP
from torchao.quantization import Int8DynamicActivationInt4WeightConfig
from torchao.quantization.qat import QATConfig
from tqdm import tqdm
from vit import ViTConfig

from mjepa_cifar10.data import get_train_dataloader, get_val_dataloader
from mjepa_cifar10.pretrain import CIFAR10MJEPA, train


SEED: Final = 0


def ddp_setup() -> None:
    """Initialize distributed training process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        logging.info("Initialized DDP")
    else:
        logging.info("DDP already initialized")


def ddp_cleanup() -> None:
    """Clean up distributed training process group."""
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
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint to load")
    return parser.parse_args()


def main(args: Namespace) -> None:
    torch.random.manual_seed(SEED)
    if not (config_path := Path(args.config)).is_file():
        raise FileNotFoundError(config_path)
    config = yaml.full_load(config_path.read_text())

    # Extract instantiated dataclasses from config
    backbone_config = config["backbone"]
    jepa_config = config["jepa"]
    optimizer_config = config["optimizer"]
    trainer_config = config["trainer"]
    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(jepa_config, JEPAConfig)
    assert isinstance(optimizer_config, OptimizerConfig)
    assert isinstance(trainer_config, TrainerConfig)
    if args.log_dir and not args.log_dir.is_dir():
        raise NotADirectoryError(args.log_dir)

    # Determine distributed training parameters
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        ddp_setup()

    # Instantiate other model elements and move to device
    device = torch.device("cuda", local_rank)
    qat_config = QATConfig(Int8DynamicActivationInt4WeightConfig())
    backbone = backbone_config.instantiate(device=device)
    backbone.apply_quantization(mlp_quantization_config=qat_config)
    predictor = CrossAttentionPredictor(backbone, jepa_config.predictor_depth, device=device)
    jepa = CIFAR10MJEPA(jepa_config, backbone, predictor)

    # Wrap in DDP for distributed training
    if world_size > 1:
        ddp_setup()
        jepa = DDP(jepa, device_ids=[local_rank])
        unwrapped_jepa = jepa.module
    else:
        unwrapped_jepa = jepa

    # Instantiate dataloaders
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
    train_dataloader = train_dataloader_fn(unwrapped_jepa.img_size, trainer_config.batch_size)

    # Instantiate optimizer and scheduler
    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    optimizer, scheduler = optimizer_config.instantiate(jepa, total_steps=total_steps)

    # Initialize wandb
    if is_rank_zero():
        wandb.init(
            project="mjepa-cifar10",
            name=args.name,
            dir=args.log_dir,
            config={
                "backbone": backbone_config.__dict__,
                "jepa": jepa_config.__dict__,
                "optimizer": optimizer_config.__dict__,
                "trainer": trainer_config.__dict__,
            },
            tags=("pretrain", config_path.stem),
            group="pretrain",
        )

    ignore_warnings()
    exit_code = 0
    try:
        with tqdm.external_write_mode():
            logging.info(f"Starting training with local rank: {local_rank}, world size: {world_size}")
        train(
            jepa,
            train_dataloader_fn,
            val_dataloader_fn,
            optimizer,
            scheduler,
            trainer_config,
        )
    except Exception as e:
        logging.error(f"Error in training: {e}")
        exit_code = 1
        raise e
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
