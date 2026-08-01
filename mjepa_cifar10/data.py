import hashlib
import math
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
DEFAULT_GLOBAL_CROP_SCALE: Final[tuple[float, float]] = (0.75, 1.0)
DEFAULT_LOCAL_CROP_SCALE: Final[tuple[float, float]] = (0.30, 0.75)
TRAIN_CROP_RATIO: Final[tuple[float, float]] = (0.75, 1.33)


@dataclass(frozen=True)
class StratifiedSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def _validate_crop_scale(name: str, scale: tuple[float, float]) -> None:
    if len(scale) != 2 or not all(math.isfinite(value) for value in scale):
        raise ValueError(f"{name} must contain two finite values")
    minimum, maximum = scale
    if not 0 < minimum <= maximum <= 1:
        raise ValueError(f"{name} must satisfy 0 < minimum <= maximum <= 1")


@dataclass(frozen=True)
class MultiCropConfig:
    """Training-only independently augmented views at one model input resolution."""

    global_views: int = 1
    local_views: int = 0
    global_scale: tuple[float, float] = DEFAULT_GLOBAL_CROP_SCALE
    local_scale: tuple[float, float] = DEFAULT_LOCAL_CROP_SCALE

    def __post_init__(self) -> None:
        if isinstance(self.global_views, bool) or not isinstance(self.global_views, int) or self.global_views <= 0:
            raise ValueError("global_views must be a positive integer")
        if isinstance(self.local_views, bool) or not isinstance(self.local_views, int) or self.local_views < 0:
            raise ValueError("local_views must be a non-negative integer")
        object.__setattr__(self, "global_scale", tuple(float(value) for value in self.global_scale))
        object.__setattr__(self, "local_scale", tuple(float(value) for value in self.local_scale))
        _validate_crop_scale("global_scale", self.global_scale)
        _validate_crop_scale("local_scale", self.local_scale)

    @property
    def total_views(self) -> int:
        return self.global_views + self.local_views

    @property
    def enabled(self) -> bool:
        return self.total_views > 1


class MultiCropTransform:
    """Apply independent global and local augmentations and stack their results."""

    def __init__(
        self,
        global_transform: Compose,
        local_transform: Compose,
        config: MultiCropConfig,
    ) -> None:
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.config = config

    def __call__(self, image: Any) -> torch.Tensor:
        views = [self.global_transform(image) for _ in range(self.config.global_views)]
        views.extend(self.local_transform(image) for _ in range(self.config.local_views))
        return torch.stack(views)


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


def _get_train_view_transform(size: Sequence[int], crop_scale: tuple[float, float]) -> Compose:
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomInvert(p=0.1),
            RandomResizedCrop(size=size, scale=crop_scale, ratio=TRAIN_CROP_RATIO),
            RandomApply([RandomRotation(degrees=cast(Any, 15))], p=0.25),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            RandomGrayscale(p=0.1),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=MEAN, std=STD),
        ]
    )


def get_train_transforms(
    size: Sequence[int],
    multi_crop_config: MultiCropConfig | None = None,
) -> Compose | MultiCropTransform:
    selected_config = multi_crop_config or MultiCropConfig()
    global_transform = _get_train_view_transform(size, selected_config.global_scale)
    if not selected_config.enabled:
        return global_transform
    local_transform = _get_train_view_transform(size, selected_config.local_scale)
    return MultiCropTransform(global_transform, local_transform, selected_config)


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
    multi_crop_config: MultiCropConfig | None = None,
) -> DataLoader:
    transforms = get_train_transforms(size, multi_crop_config=multi_crop_config)
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


def get_probe_train_dataloader(
    size: Sequence[int],
    batch_size: int,
    root: Path,
    num_workers: int,
) -> DataLoader:
    """Return the fixed training split with deterministic evaluation transforms."""
    transforms = get_val_transforms(size)
    persistent_workers = num_workers > 0
    base_dataset = CIFAR10(root=root, train=True, transform=transforms, download=True)
    split = build_stratified_split_indices(base_dataset.targets)
    dataset = Subset(base_dataset, split.train_indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
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
