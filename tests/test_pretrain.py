from types import SimpleNamespace

import torch

from mjepa_cifar10.pretrain import get_scheduler_last_lr


def test_get_scheduler_last_lr_returns_first_learning_rate() -> None:
    scheduler = SimpleNamespace(get_last_lr=lambda: [0.2, 0.1])

    assert get_scheduler_last_lr(scheduler) == 0.2
