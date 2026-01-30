from pathlib import Path
from typing import Any, Final, Sequence, cast

import torch
from torch.utils.data import DataLoader, DistributedSampler
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
) -> DataLoader:
    transforms = get_train_transforms(size)
    dataset = CIFAR10(root=root, train=True, transform=transforms, download=True)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )


def get_val_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
) -> DataLoader:
    transforms = get_val_transforms(size)
    dataset = CIFAR10(root=root, train=False, transform=transforms, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=True,
    )
