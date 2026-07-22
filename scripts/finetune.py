import json
import logging
import os
import shutil
import socket
import sys
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Final

import torch
import torch.distributed as dist
import wandb
import yaml
from mjepa.optimizer import OptimizerConfig
from mjepa.trainer import (
    TrainerConfig,
    calculate_total_steps,
    ignore_warnings,
    is_rank_zero,
    seed_everything,
    setup_logdir,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from vit import ViTConfig

from mjepa_cifar10.data import cifar10_split_fingerprint, get_test_dataloader, get_train_dataloader, get_val_dataloader
from mjepa_cifar10.experiment import write_run_metadata
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
    parser.add_argument("--exact-log-dir", type=Path, default=None, help="Existing managed run directory")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank / device")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to backbone safetensors checkpoint")
    parser.add_argument("--seed", type=int, default=SEED, help="Training initialization seed")
    parser.add_argument("--shots-per-class", type=int, choices=(10, 100), default=None)
    parser.add_argument("--subset-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="mjepa-cifar10")
    parser.add_argument("--wandb-group", type=str, default="finetune")
    parser.add_argument("--study-id", type=str, default=None)
    parser.add_argument("--model-class", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--provenance-file", type=Path, default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the official CIFAR-10 test set; reserved for confirmed baseline/winner runs",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    seed_everything(args.seed)
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
    if args.exact_log_dir and not args.exact_log_dir.is_dir():
        raise NotADirectoryError(args.exact_log_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or args.local_rank)
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        ddp_setup()

    if args.exact_log_dir is not None:
        setup_logdir(None, None)
        run_log_dir = args.exact_log_dir.resolve() if is_rank_zero() else None
        if is_rank_zero():
            assert run_log_dir is not None
        if run_log_dir is not None and not (run_log_dir / "config.yaml").exists():
            shutil.copyfile(config_path, run_log_dir / "config.yaml")
    else:
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
        shots_per_class=args.shots_per_class,
        subset_seed=args.subset_seed,
    )
    val_dataloader_fn = partial(
        get_val_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
    )
    test_dataloader_fn = partial(
        get_test_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
    )
    train_dataloader = train_dataloader_fn(unwrapped_model.img_size, trainer_config.batch_size)

    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    optimizer, scheduler = optimizer_config.instantiate(model, total_steps=total_steps)

    if is_rank_zero():
        external_provenance = json.loads(args.provenance_file.read_text()) if args.provenance_file else {}
        provenance_config = {
            "provenance/seed": args.seed,
            "provenance/subset_seed": args.subset_seed,
            "provenance/shots_per_class": args.shots_per_class,
            "provenance/study_id": args.study_id,
            "provenance/model_class": args.model_class,
            "provenance/variant": args.variant,
            "provenance/physical_gpu": args.physical_gpu,
            "provenance/hostname": socket.gethostname(),
            "provenance/command": list(sys.argv),
            "provenance/config": str(config_path.resolve()),
            "provenance/dataset_split_hash": cifar10_split_fingerprint(args.data),
            "provenance/local_weight_disposition": "retained",
            "provenance/lockfile_sha256": external_provenance.get("lockfile_sha256"),
        }
        for repository in ("parent", "mjepa", "vit"):
            for key, value in external_provenance.get(repository, {}).items():
                provenance_config[f"provenance/{repository}_{key}"] = value
        initialized_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            dir=run_log_dir,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config={
                "backbone": backbone_config.__dict__,
                "optimizer": optimizer_config.__dict__,
                "trainer": trainer_config.__dict__,
                "checkpoint": str(args.checkpoint),
                **provenance_config,
            },
            tags=("finetune", config_path.stem),
            group=args.wandb_group,
        )
        write_run_metadata(
            run_log_dir,
            {
                "wandb_run_id": initialized_run.id,
                "wandb_url": initialized_run.url,
                "config": str(config_path.resolve()),
                "command": list(sys.argv),
                "provenance": provenance_config,
                "local_weight_disposition": "retained",
                "model_class": args.model_class,
            },
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
            test_dataloader_fn=test_dataloader_fn if args.evaluate_test else None,
            wandb_run_id=args.wandb_run_id or (wandb.run.id if wandb.run is not None else None),
            output_dir=run_log_dir,
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
