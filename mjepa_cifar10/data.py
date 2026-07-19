import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence, cast

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    Normalize,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomInvert,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToDtype,
    ToImage,
)


MEAN: Final = (0.4914, 0.4822, 0.4465)
STD: Final = (0.2470, 0.2434, 0.2616)
PIN_MEMORY: Final[bool] = True
NUM_CLASSES: Final[int] = 10
SPLIT_SEED: Final[int] = 0
VALIDATION_PER_CLASS: Final[int] = 500
SPLIT_VERSION: Final[str] = "cifar10-stratified-sha256-v1"


@dataclass(frozen=True)
class StratifiedSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def _stable_index_key(index: int, split_seed: int) -> bytes:
    value = f"{SPLIT_VERSION}:{split_seed}:{index}".encode()
    return hashlib.sha256(value).digest()


def build_stratified_split_indices(
    targets: Sequence[int],
    validation_per_class: int = VALIDATION_PER_CLASS,
    split_seed: int = SPLIT_SEED,
) -> StratifiedSplit:
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")

    class_to_indices: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        class_to_indices.setdefault(int(target), []).append(index)

    if sorted(class_to_indices) != list(range(NUM_CLASSES)):
        raise ValueError(f"expected CIFAR-10 classes 0-{NUM_CLASSES - 1}, got {sorted(class_to_indices)}")

    validation_indices: list[int] = []
    train_indices: list[int] = []
    for class_index in range(NUM_CLASSES):
        indices = class_to_indices[class_index]
        if validation_per_class >= len(indices):
            raise ValueError(
                f"validation split requests {validation_per_class} examples for class {class_index}, "
                f"but only {len(indices)} are available"
            )
        ordered_indices = sorted(indices, key=lambda index: _stable_index_key(index, split_seed))
        validation_indices.extend(ordered_indices[:validation_per_class])
        train_indices.extend(ordered_indices[validation_per_class:])

    return StratifiedSplit(tuple(sorted(train_indices)), tuple(sorted(validation_indices)))


def split_fingerprint(split: StratifiedSplit) -> str:
    digest = hashlib.sha256()
    digest.update(SPLIT_VERSION.encode())
    for index in split.validation_indices:
        digest.update(index.to_bytes(4, byteorder="big", signed=False))
    return digest.hexdigest()


def cifar10_split_fingerprint(root: Path) -> str:
    """Return the fixed holdout hash without applying transforms."""
    dataset = CIFAR10(root=root, train=True, download=False)
    return split_fingerprint(build_stratified_split_indices(dataset.targets))


def restrict_dataset_to_few_shot(
    dataset: Any,
    shots_per_class: int | None,
    subset_seed: int,
    candidate_indices: Sequence[int] | None = None,
) -> Any | Subset[Any]:
    if shots_per_class is None:
        return dataset if candidate_indices is None else Subset(dataset, list(candidate_indices))
    if shots_per_class <= 0:
        raise ValueError("shots_per_class must be positive")

    selected_pool = range(len(dataset)) if candidate_indices is None else candidate_indices
    class_to_indices: dict[int, list[int]] = {class_index: [] for class_index in range(NUM_CLASSES)}
    for index in selected_pool:
        class_to_indices[int(dataset.targets[index])].append(index)

    selected_indices: list[int] = []
    for class_index in range(NUM_CLASSES):
        indices = class_to_indices[class_index]
        if shots_per_class > len(indices):
            raise ValueError(
                f"few-shot split requests {shots_per_class} examples for class {class_index}, "
                f"but only {len(indices)} are available"
            )
        ordered_indices = sorted(indices, key=lambda index: _stable_index_key(index, subset_seed))
        selected_indices.extend(ordered_indices[:shots_per_class])

    return Subset(dataset, sorted(selected_indices))


def get_train_transforms(size: Sequence[int]) -> Compose:
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomInvert(p=0.1),
            RandomResizedCrop(size=size, scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            RandomApply([RandomRotation(degrees=cast(Any, 15))], p=0.25),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            RandomGrayscale(p=0.1),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=MEAN, std=STD),
        ]
    )


def get_val_transforms(size: Sequence[int]) -> Compose:
    return Compose(
        [
            Resize(size=size),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=MEAN, std=STD),
        ]
    )


def get_train_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
    local_rank: int,
    world_size: int,
    shots_per_class: int | None = None,
    subset_seed: int = SPLIT_SEED,
) -> DataLoader:
    transforms = get_train_transforms(size)
    persistent_workers = num_workers > 0
    # Only rank 0 downloads to avoid race conditions
    if world_size > 1:
        if local_rank == 0:
            CIFAR10(root=root, train=True, download=True)
        dist.barrier()
    base_dataset = CIFAR10(root=root, train=True, transform=transforms, download=(world_size == 1))
    split = build_stratified_split_indices(base_dataset.targets)
    dataset = restrict_dataset_to_few_shot(
        base_dataset,
        shots_per_class=shots_per_class,
        subset_seed=subset_seed,
        candidate_indices=split.train_indices,
    )
    drop_last = len(dataset) >= batch_size
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=drop_last,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            sampler=sampler,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )


def get_val_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
) -> DataLoader:
    transforms = get_val_transforms(size)
    persistent_workers = num_workers > 0
    # Only rank 0 downloads to avoid race conditions
    if dist.is_initialized():
        if dist.get_rank() == 0:
            CIFAR10(root=root, train=True, download=True)
        dist.barrier()
    base_dataset = CIFAR10(root=root, train=True, transform=transforms, download=not dist.is_initialized())
    split = build_stratified_split_indices(base_dataset.targets)
    dataset = Subset(base_dataset, split.validation_indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
        persistent_workers=persistent_workers,
    )


def get_test_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
) -> DataLoader:
    transforms = get_val_transforms(size)
    persistent_workers = num_workers > 0
    if dist.is_initialized():
        if dist.get_rank() == 0:
            CIFAR10(root=root, train=False, download=True)
        dist.barrier()
    dataset = CIFAR10(root=root, train=False, transform=transforms, download=not dist.is_initialized())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
