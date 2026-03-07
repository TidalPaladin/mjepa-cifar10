from pathlib import Path

import mjepa_cifar10.data as data


BATCH_SIZE = 32
IMG_SIZE = (32, 32)
ROOT = Path("/tmp/cifar10")


def test_get_train_dataloader_ddp_enables_persistent_workers_and_pin_memory(mocker) -> None:
    dataset = object()
    sampler = object()
    dataloader = object()
    cifar10_mock = mocker.patch.object(data, "CIFAR10", side_effect=[dataset, dataset])
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
    distributed_sampler_mock.assert_called_once_with(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
    )
    barrier_mock.assert_called_once_with()
    dataloader_mock.assert_called_once_with(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        sampler=sampler,
    )


def test_get_train_dataloader_single_gpu_disables_persistent_workers_with_zero_workers(mocker) -> None:
    dataset = object()
    dataloader = object()
    mocker.patch.object(data, "CIFAR10", return_value=dataset)
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
    dataloader_mock.assert_called_once_with(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )


def test_get_val_dataloader_disables_persistent_workers_with_zero_workers(mocker) -> None:
    dataset = object()
    dataloader = object()
    mocker.patch.object(data.dist, "is_initialized", return_value=False)
    mocker.patch.object(data, "CIFAR10", return_value=dataset)
    dataloader_mock = mocker.patch.object(data, "DataLoader", return_value=dataloader)

    result = data.get_val_dataloader(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        root=ROOT,
        num_workers=0,
    )

    assert result is dataloader
    dataloader_mock.assert_called_once_with(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )
