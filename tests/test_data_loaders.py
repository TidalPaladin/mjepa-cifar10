from pathlib import Path

import pytest
from torch.utils.data import Subset

import mjepa_cifar10.data as data


BATCH_SIZE = 32
IMG_SIZE = (32, 32)
ROOT = Path("/tmp/cifar10")
CLASSES = 10
EXAMPLES_PER_CLASS = 20
VALIDATION_PER_CLASS = 5


class DatasetWithTargets:
    def __init__(self) -> None:
        self.targets = [class_index for class_index in range(CLASSES) for _ in range(EXAMPLES_PER_CLASS)]

    def __len__(self) -> int:
        return len(self.targets)


def test_build_stratified_split_is_deterministic_balanced_and_disjoint() -> None:
    targets = DatasetWithTargets().targets

    first = data.build_stratified_split_indices(
        targets,
        validation_per_class=VALIDATION_PER_CLASS,
        split_seed=0,
    )
    second = data.build_stratified_split_indices(
        targets,
        validation_per_class=VALIDATION_PER_CLASS,
        split_seed=0,
    )

    assert first == second
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    assert sorted((*first.train_indices, *first.validation_indices)) == list(range(len(targets)))
    for class_index in range(CLASSES):
        assert sum(targets[index] == class_index for index in first.validation_indices) == VALIDATION_PER_CLASS


def test_restrict_dataset_to_few_shot_selects_exact_count_per_class() -> None:
    dataset = DatasetWithTargets()

    subset = data.restrict_dataset_to_few_shot(dataset, shots_per_class=10, subset_seed=1)

    assert isinstance(subset, Subset)
    assert len(subset) == CLASSES * 10
    for class_index in range(CLASSES):
        assert sum(dataset.targets[index] == class_index for index in subset.indices) == 10


def test_restrict_dataset_to_few_shot_rejects_too_many_shots() -> None:
    dataset = DatasetWithTargets()

    with pytest.raises(ValueError, match="only 20 are available"):
        data.restrict_dataset_to_few_shot(dataset, shots_per_class=21, subset_seed=0)


def test_get_train_dataloader_ddp_enables_persistent_workers_and_pin_memory(mocker) -> None:
    download_only_dataset = object()
    base_dataset = DatasetWithTargets()
    split = data.StratifiedSplit(tuple(range(100)), tuple(range(100, 200)))
    sampler = object()
    dataloader = object()
    cifar10_mock = mocker.patch.object(data, "CIFAR10", side_effect=[download_only_dataset, base_dataset])
    mocker.patch.object(data, "build_stratified_split_indices", return_value=split)
    distributed_sampler_mock = mocker.patch.object(data, "DistributedSampler", return_value=sampler)
    barrier_mock = mocker.patch.object(data.dist, "barrier")
    dataloader_mock = mocker.patch.object(data, "DataLoader", return_value=dataloader)

    result = data.get_train_dataloader(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        root=ROOT,
        num_workers=4,
        local_rank=0,
        world_size=2,
    )

    assert result is dataloader
    assert cifar10_mock.call_count == 2
    subset = distributed_sampler_mock.call_args.args[0]
    assert isinstance(subset, Subset)
    assert subset.dataset is base_dataset
    assert tuple(subset.indices) == split.train_indices
    distributed_sampler_mock.assert_called_once_with(
        subset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
    )
    barrier_mock.assert_called_once_with()
    dataloader_mock.assert_called_once_with(
        subset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        sampler=sampler,
    )


def test_get_train_dataloader_single_gpu_disables_persistent_workers_with_zero_workers(mocker) -> None:
    base_dataset = DatasetWithTargets()
    split = data.StratifiedSplit(tuple(range(100)), tuple(range(100, 200)))
    dataloader = object()
    mocker.patch.object(data, "CIFAR10", return_value=base_dataset)
    mocker.patch.object(data, "build_stratified_split_indices", return_value=split)
    dataloader_mock = mocker.patch.object(data, "DataLoader", return_value=dataloader)

    result = data.get_train_dataloader(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        root=ROOT,
        num_workers=0,
        local_rank=0,
        world_size=1,
    )

    assert result is dataloader
    subset = dataloader_mock.call_args.args[0]
    assert isinstance(subset, Subset)
    assert subset.dataset is base_dataset
    assert tuple(subset.indices) == split.train_indices
    dataloader_mock.assert_called_once_with(
        subset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )


def test_get_val_dataloader_disables_persistent_workers_with_zero_workers(mocker) -> None:
    base_dataset = DatasetWithTargets()
    split = data.StratifiedSplit(tuple(range(100)), tuple(range(100, 200)))
    dataloader = object()
    mocker.patch.object(data.dist, "is_initialized", return_value=False)
    mocker.patch.object(data, "CIFAR10", return_value=base_dataset)
    mocker.patch.object(data, "build_stratified_split_indices", return_value=split)
    dataloader_mock = mocker.patch.object(data, "DataLoader", return_value=dataloader)

    result = data.get_val_dataloader(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        root=ROOT,
        num_workers=0,
    )

    assert result is dataloader
    subset = dataloader_mock.call_args.args[0]
    assert isinstance(subset, Subset)
    assert subset.dataset is base_dataset
    assert tuple(subset.indices) == split.validation_indices
    dataloader_mock.assert_called_once_with(
        subset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )


def test_get_train_dataloader_keeps_small_few_shot_subset_nonempty(mocker) -> None:
    base_dataset = DatasetWithTargets()
    split = data.StratifiedSplit(tuple(range(200)), ())
    dataloader = object()
    mocker.patch.object(data, "CIFAR10", return_value=base_dataset)
    mocker.patch.object(data, "build_stratified_split_indices", return_value=split)
    dataloader_mock = mocker.patch.object(data, "DataLoader", return_value=dataloader)

    result = data.get_train_dataloader(
        size=IMG_SIZE,
        batch_size=512,
        root=ROOT,
        num_workers=0,
        local_rank=0,
        world_size=1,
        shots_per_class=10,
        subset_seed=0,
    )

    assert result is dataloader
    subset = dataloader_mock.call_args.args[0]
    assert len(subset) == 100
    assert dataloader_mock.call_args.kwargs["drop_last"] is False
