import json
import logging
import math
import os
import shutil
import socket
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Final, NamedTuple

import torch
import torch.distributed as dist
import yaml
from mjepa.jepa import ADALN_BLIND_CLS_PREDICTION_MODE, CrossAttentionPredictor, JEPAConfig
from mjepa.optimizer import OptimizerConfig, OptimizerLike, SchedulerLike
from mjepa.trainer import (
    CheckpointMetadata,
    TrainerConfig,
    calculate_total_steps,
    ignore_warnings,
    is_rank_zero,
    load_checkpoint,
    load_checkpoint_metadata,
    seed_everything,
    setup_logdir,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from vit import ViTConfig

import wandb
from mjepa_cifar10.data import cifar10_split_fingerprint, get_test_dataloader, get_train_dataloader, get_val_dataloader
from mjepa_cifar10.experiment import append_metric_record, write_run_metadata
from mjepa_cifar10.pretrain import (
    CIFAR10MJEPA,
    DEFAULT_CLS_GLOBAL_TARGET_LOSS_WEIGHT,
    FirstCycleCallback,
    train,
)
from mjepa_cifar10.research.cls_path_benchmark import benchmark_cls_prediction_path, write_cls_path_benchmark
from mjepa_cifar10.research.lifecycle_events import RunLifecycleReporter
from mjepa_cifar10.research.runtime import (
    LIFECYCLE_ATTEMPT_ENVIRONMENT_VARIABLE,
    LIFECYCLE_RUN_ENVIRONMENT_VARIABLE,
    LIFECYCLE_STUDY_ENVIRONMENT_VARIABLE,
    LIFECYCLE_THREAD_ENVIRONMENT_VARIABLE,
)


SEED: Final = 0


class ResumeState(NamedTuple):
    step: int
    epoch: int
    elapsed_seconds: float
    wandb_run_id: str | None


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
    parser.add_argument("--exact-log-dir", type=Path, default=None, help="Existing managed run directory")
    parser.add_argument("--local-rank", type=int, default=1, help="Local rank / device")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint to load")
    parser.add_argument("--seed", type=int, default=SEED, help="Training and data seed")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="W&B run ID for managed launch or resume")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="mjepa-cifar10")
    parser.add_argument("--wandb-group", type=str, default="pretrain", help="W&B run group")
    parser.add_argument("--study-id", type=str, default=None, help="Managed research study ID")
    parser.add_argument("--model-class", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None, help="Managed research variant ID")
    parser.add_argument("--physical-gpu", type=int, default=None, help="Physical GPU recorded in provenance")
    parser.add_argument("--provenance-file", type=Path, default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the official CIFAR-10 test set; reserved for confirmed baseline/winner runs",
    )
    return parser.parse_args()


def instantiate_jepa(backbone_config: ViTConfig, jepa_config: JEPAConfig, device: torch.device) -> CIFAR10MJEPA:
    backbone = backbone_config.instantiate(device=device)
    predictor = CrossAttentionPredictor(
        backbone,
        jepa_config.predictor_depth,
        device=device,
        attention_mode=jepa_config.predictor_attention_mode,
        cls_prediction_mode=jepa_config.cls_prediction_mode,
        disable_predictor_regularizers=jepa_config.disable_predictor_regularizers,
    )
    return CIFAR10MJEPA(jepa_config, backbone, predictor)


def validate_cls_global_target_configuration(
    backbone_config: ViTConfig,
    jepa_config: JEPAConfig,
    loss_weight: float,
) -> None:
    if not math.isfinite(loss_weight) or loss_weight < 0:
        raise ValueError("cls_global_target_loss_weight must be a finite non-negative float")
    if loss_weight == 0:
        return
    if backbone_config.num_cls_tokens != 1:
        raise ValueError("CLS global-target loss requires exactly one student CLS token")
    if jepa_config.cls_prediction_mode != ADALN_BLIND_CLS_PREDICTION_MODE:
        raise ValueError("CLS global-target loss requires cls_prediction_mode='adaln_blind'")


def restore_pretraining_checkpoint(
    checkpoint: Path,
    metadata: CheckpointMetadata,
    jepa: CIFAR10MJEPA,
    optimizer: OptimizerLike,
    scheduler: SchedulerLike,
    requested_wandb_run_id: str | None,
) -> ResumeState:
    if requested_wandb_run_id and metadata.wandb_run_id and requested_wandb_run_id != metadata.wandb_run_id:
        raise ValueError(
            f"requested W&B run ID {requested_wandb_run_id!r} does not match checkpoint "
            f"run ID {metadata.wandb_run_id!r}"
        )
    step, epoch = load_checkpoint(
        checkpoint,
        jepa.student,
        jepa.predictor,
        jepa.teacher,
        optimizer,
        scheduler,
    )
    return ResumeState(
        step,
        epoch,
        metadata.elapsed_seconds,
        requested_wandb_run_id or metadata.wandb_run_id,
    )


def apply_checkpoint_image_size(backbone_config: ViTConfig, metadata: CheckpointMetadata) -> ViTConfig:
    if metadata.img_size is None:
        return backbone_config
    return replace(backbone_config, img_size=list(metadata.img_size))


def should_benchmark_cls_prediction_path(checkpoint: Path | None) -> bool:
    """Preserve the immutable launch benchmark when restoring a checkpoint."""
    return checkpoint is None


def build_managed_lifecycle_reporter(
    args: Namespace,
    run_log_dir: Path | None,
    environment: Mapping[str, str] | None = None,
) -> RunLifecycleReporter | None:
    """Create a reporter only for a supervisor-bound managed run."""
    selected_environment = os.environ if environment is None else environment
    if args.study_id is None or run_log_dir is None:
        return None
    study_id = selected_environment.get(LIFECYCLE_STUDY_ENVIRONMENT_VARIABLE)
    run_id = selected_environment.get(LIFECYCLE_RUN_ENVIRONMENT_VARIABLE)
    attempt_text = selected_environment.get(LIFECYCLE_ATTEMPT_ENVIRONMENT_VARIABLE)
    if study_id is None and run_id is None and attempt_text is None:
        return None
    if study_id != args.study_id or run_id != args.name:
        raise ValueError("managed lifecycle environment does not match the requested run")
    assert study_id is not None
    assert run_id is not None
    try:
        attempt = int(attempt_text or "")
    except ValueError as error:
        raise ValueError("managed lifecycle attempt must be a positive integer") from error
    if attempt < 1:
        raise ValueError("managed lifecycle attempt must be a positive integer")
    return RunLifecycleReporter(
        run_dir=run_log_dir,
        study_id=study_id,
        run_id=run_id,
        attempt=attempt,
        originating_thread_id=selected_environment.get(LIFECYCLE_THREAD_ENVIRONMENT_VARIABLE),
    )


def main(args: Namespace) -> None:
    seed_everything(args.seed)
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
    cls_global_target_loss_weight = float(
        config.get("cls_global_target_loss_weight", DEFAULT_CLS_GLOBAL_TARGET_LOSS_WEIGHT)
    )
    validate_cls_global_target_configuration(backbone_config, jepa_config, cls_global_target_loss_weight)
    if args.log_dir and not args.log_dir.is_dir():
        raise NotADirectoryError(args.log_dir)
    if args.exact_log_dir and not args.exact_log_dir.is_dir():
        raise NotADirectoryError(args.exact_log_dir)

    # Determine distributed training parameters
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or args.local_rank)
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        ddp_setup()

    checkpoint_metadata = None
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        checkpoint_metadata = load_checkpoint_metadata(args.checkpoint)
        backbone_config = apply_checkpoint_image_size(backbone_config, checkpoint_metadata)
        setup_logdir(None, None)
        run_log_dir = args.checkpoint.resolve().parent if is_rank_zero() else None
        if args.exact_log_dir and run_log_dir != args.exact_log_dir.resolve():
            raise ValueError("resume checkpoint must be inside --exact-log-dir")
    elif args.exact_log_dir is not None:
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

    # Instantiate other model elements and move to device
    device = torch.device("cuda", local_rank)
    jepa = instantiate_jepa(backbone_config, jepa_config, device)

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
    test_dataloader_fn = partial(
        get_test_dataloader,
        root=args.data,
        num_workers=trainer_config.num_workers,
    )
    train_dataloader = train_dataloader_fn(unwrapped_jepa.img_size, trainer_config.batch_size)

    # Instantiate optimizer and scheduler
    total_steps = calculate_total_steps(
        train_dataloader, trainer_config.num_epochs, trainer_config.accumulate_grad_batches
    )
    optimizer, scheduler = optimizer_config.instantiate(jepa, total_steps=total_steps)

    initial_step = 0
    last_epoch = -1
    elapsed_seconds = 0.0
    wandb_run_id = args.wandb_run_id
    if args.checkpoint is not None:
        assert checkpoint_metadata is not None
        resume_state = restore_pretraining_checkpoint(
            args.checkpoint,
            checkpoint_metadata,
            unwrapped_jepa,
            optimizer,
            scheduler,
            wandb_run_id,
        )
        initial_step = resume_state.step
        last_epoch = resume_state.epoch
        elapsed_seconds = resume_state.elapsed_seconds
        wandb_run_id = resume_state.wandb_run_id

    # Initialize wandb
    if is_rank_zero():
        external_provenance = json.loads(args.provenance_file.read_text()) if args.provenance_file else {}
        provenance_config = {
            "provenance/seed": args.seed,
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
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            config={
                "backbone": backbone_config.__dict__,
                "jepa": jepa_config.__dict__,
                "optimizer": optimizer_config.__dict__,
                "trainer": trainer_config.__dict__,
                "cls_global_target_loss_weight": cls_global_target_loss_weight,
                **provenance_config,
            },
            tags=("pretrain", config_path.stem),
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
                "cls_global_target_loss_weight": cls_global_target_loss_weight,
            },
        )
        if should_benchmark_cls_prediction_path(args.checkpoint):
            cls_path_benchmark = benchmark_cls_prediction_path(unwrapped_jepa)
            cls_path_metrics = cls_path_benchmark.to_metrics()
            wandb.config.update({"cls_path_benchmark": cls_path_benchmark.to_dict()})
            wandb.log(cls_path_metrics, step=initial_step)
            append_metric_record(run_log_dir, initial_step, cls_path_metrics)
            if run_log_dir is not None:
                write_cls_path_benchmark(run_log_dir / "cls-path-benchmark.json", cls_path_benchmark)
    if dist.is_initialized():
        dist.barrier()

    ignore_warnings()
    lifecycle_reporter = build_managed_lifecycle_reporter(args, run_log_dir)
    first_cycle_callback: FirstCycleCallback | None = None
    if lifecycle_reporter is not None and run_log_dir is not None:
        checkpoint_path = run_log_dir / "checkpoint.pt"

        def report_first_cycle(epoch: int, optimizer_step: int, active_seconds: float) -> object:
            return lifecycle_reporter.first_cycle(
                epoch,
                optimizer_step,
                active_seconds,
                checkpoint_path=checkpoint_path,
            )

        first_cycle_callback = report_first_cycle

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
            test_dataloader_fn=test_dataloader_fn if args.evaluate_test else None,
            last_epoch=last_epoch,
            initial_step=initial_step,
            elapsed_seconds_offset=elapsed_seconds,
            wandb_run_id=wandb_run_id or (wandb.run.id if wandb.run is not None else None),
            output_dir=run_log_dir,
            max_grad_norm=optimizer_config.max_grad_norm,
            cls_global_target_loss_weight=cls_global_target_loss_weight,
            progress_callback=lifecycle_reporter.progress if lifecycle_reporter is not None else None,
            first_cycle_callback=first_cycle_callback,
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
