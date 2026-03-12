from collections.abc import Sequence
from pathlib import Path

import pytest
import safetensors.torch as st
import torch
from mjepa import ResolutionConfig, TrainerConfig
from mjepa.optimizer import OptimizerConfig
from torch.utils.data import DataLoader, TensorDataset
from vit import AttentivePoolHeadConfig, HeadConfig, ViTConfig, ViTFeatures

import mjepa_cifar10.finetune as finetune_module
from mjepa_cifar10.finetune import (
    GRAD_CLIP_TRIGGER_PCT_KEY,
    CIFAR10FineTuner,
    build_train_log_dict,
    build_val_log_dict,
    load_backbone_checkpoint,
    train,
    validate_finetune_config,
)


HIDDEN_SIZE = 8
NUM_REGISTER_TOKENS = 2
NUM_CLS_TOKENS = 2
NUM_VISUAL_TOKENS = 4
BATCH_SIZE = 2
NUM_CLASSES = 10


class RecordingHead(torch.nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.clone()
        if x.ndim != 2:
            raise AssertionError(f"expected a pooled embedding, got shape={tuple(x.shape)}")
        return x[:, : self.out_features]


class RecordingScheduler:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, epoch: int | None = None) -> None:
        _ = epoch

    def state_dict(self) -> dict[str, float]:
        return {"lr": self.lr}

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        self.lr = state_dict["lr"]

    def get_last_lr(self) -> list[float]:
        return [self.lr]


def make_backbone_config(
    *,
    num_cls_tokens: int,
    head_config: HeadConfig | AttentivePoolHeadConfig | None = None,
    img_size: list[int] | None = None,
) -> ViTConfig:
    return ViTConfig(
        in_channels=3,
        hidden_size=HIDDEN_SIZE,
        patch_size=[4, 4],
        img_size=img_size or [8, 8],
        depth=1,
        num_attention_heads=2,
        ffn_hidden_size=16,
        num_register_tokens=NUM_REGISTER_TOKENS,
        num_cls_tokens=num_cls_tokens,
        dtype=torch.float32,
        heads={"cls": head_config} if head_config is not None else {},
    )


def make_model(
    *,
    num_cls_tokens: int,
    head_config: HeadConfig | AttentivePoolHeadConfig | None = None,
    img_size: list[int] | None = None,
) -> CIFAR10FineTuner:
    backbone = make_backbone_config(
        num_cls_tokens=num_cls_tokens,
        head_config=head_config,
        img_size=img_size,
    ).instantiate()
    return CIFAR10FineTuner(backbone)


def make_features(*, num_cls_tokens: int) -> ViTFeatures:
    total_tokens = num_cls_tokens + NUM_REGISTER_TOKENS + NUM_VISUAL_TOKENS
    dense_features = torch.arange(BATCH_SIZE * total_tokens * HIDDEN_SIZE, dtype=torch.float32).view(
        BATCH_SIZE,
        total_tokens,
        HIDDEN_SIZE,
    )
    return ViTFeatures(dense_features, NUM_REGISTER_TOKENS, num_cls_tokens, tokenized_size=(2, 2))


def make_dataloader(size: list[int], batch_size: int) -> DataLoader:
    image_height, image_width = size
    images = torch.arange(batch_size * 2 * 3 * image_height * image_width, dtype=torch.float32).view(
        batch_size * 2,
        3,
        image_height,
        image_width,
    )
    labels = torch.tensor([0, 1] * batch_size, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_validate_finetune_config_rejects_jepa_section() -> None:
    config = {
        "backbone": make_backbone_config(
            num_cls_tokens=NUM_CLS_TOKENS, head_config=HeadConfig(out_features=NUM_CLASSES)
        ),
        "optimizer": OptimizerConfig(lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), fused=False),
        "trainer": TrainerConfig(batch_size=BATCH_SIZE, num_workers=0, num_epochs=1),
        "jepa": object(),
    }

    with pytest.raises(ValueError, match="must not include"):
        validate_finetune_config(config)


def test_load_backbone_checkpoint_requires_safetensors_suffix(tmp_path: Path) -> None:
    model = make_model(num_cls_tokens=NUM_CLS_TOKENS, head_config=HeadConfig(out_features=NUM_CLASSES))
    checkpoint_path = tmp_path / "backbone.pt"
    checkpoint_path.write_text("not a safetensors file")

    with pytest.raises(ValueError, match=r"\.safetensors"):
        load_backbone_checkpoint(checkpoint_path, model.backbone, torch.device("cpu"))


def test_load_backbone_checkpoint_restores_backbone_state(tmp_path: Path) -> None:
    source_model = make_model(num_cls_tokens=NUM_CLS_TOKENS, head_config=HeadConfig(out_features=NUM_CLASSES))
    checkpoint_path = tmp_path / "backbone.safetensors"
    st.save_file(
        {key: value for key, value in source_model.backbone.state_dict().items() if isinstance(value, torch.Tensor)},
        str(checkpoint_path),
    )

    target_model = make_model(num_cls_tokens=NUM_CLS_TOKENS, head_config=HeadConfig(out_features=NUM_CLASSES))
    for parameter in target_model.backbone.parameters():
        parameter.data.zero_()

    load_backbone_checkpoint(checkpoint_path, target_model.backbone, torch.device("cpu"))

    for key, tensor in source_model.backbone.state_dict().items():
        assert torch.equal(tensor, target_model.backbone.state_dict()[key])


def test_finetuner_forward_logits_pools_cls_tokens_before_head(mocker) -> None:
    model = make_model(num_cls_tokens=NUM_CLS_TOKENS)
    features = make_features(num_cls_tokens=NUM_CLS_TOKENS)
    head = RecordingHead(out_features=3)
    mocker.patch.object(model.backbone, "get_head", return_value=head)

    logits = model.forward_logits(features)

    assert head.last_input is not None
    assert torch.equal(head.last_input, features.cls_tokens.mean(1))
    assert logits.shape == (BATCH_SIZE, 3)


def test_finetuner_forward_logits_uses_attentive_pooling_without_cls_tokens() -> None:
    model = make_model(
        num_cls_tokens=0,
        head_config=AttentivePoolHeadConfig(
            out_features=3,
            num_attention_heads=2,
            num_queries=1,
        ),
    )
    features = make_features(num_cls_tokens=0)

    logits = model.forward_logits(features)

    assert logits.shape == (BATCH_SIZE, 3)
    assert torch.isfinite(logits).all()


def test_finetuner_forward_logits_requires_single_embedding_without_cls_tokens(mocker) -> None:
    model = make_model(num_cls_tokens=0)
    features = make_features(num_cls_tokens=0)
    mocker.patch.object(model.backbone, "get_head", return_value=torch.nn.Identity())

    with pytest.raises(ValueError, match="single embedding per sample"):
        model.forward_logits(features)


def test_build_log_dicts_keep_accuracy_as_primary_metric() -> None:
    train_loss = finetune_module.tm.MeanMetric()
    train_loss.update(1.5)
    train_acc = finetune_module.tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    train_acc.update(
        torch.tensor([[0.9, 0.1] + [0.0] * 8, [0.8, 0.2] + [0.0] * 8], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
    )
    clip_metric = finetune_module.tm.MeanMetric()
    clip_metric.update(1.0)
    scheduler = RecordingScheduler(lr=0.002)
    val_acc = finetune_module.tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    val_acc.update(
        torch.tensor([[0.7, 0.3] + [0.0] * 8, [0.1, 0.9] + [0.0] * 8], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
    )

    train_log = build_train_log_dict(
        train_loss,
        train_acc,
        scheduler,
        grad_norm_stats=(0.3, 0.7),
        train_grad_clip_trigger_pct=clip_metric,
    )
    val_log = build_val_log_dict(val_acc, epoch=3)

    assert train_log == {
        "train/loss": pytest.approx(1.5),
        "train/acc": pytest.approx(0.5),
        "train/lr": pytest.approx(0.002),
        "train/grad_norm_mean": pytest.approx(0.3),
        "train/grad_norm_max": pytest.approx(0.7),
        GRAD_CLIP_TRIGGER_PCT_KEY: pytest.approx(100.0),
    }
    assert val_log == {
        "val/acc": pytest.approx(1.0),
        "val/epoch": 3,
    }


def test_train_applies_resolution_scaling_and_logs_accuracy_metrics(mocker) -> None:
    model = make_model(
        num_cls_tokens=NUM_CLS_TOKENS,
        head_config=HeadConfig(out_features=NUM_CLASSES),
        img_size=[8, 8],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingScheduler(lr=0.1)
    trainer_config = TrainerConfig(
        batch_size=BATCH_SIZE,
        num_workers=0,
        num_epochs=1,
        accumulate_grad_batches=1,
        log_interval=1,
        check_val_every_n_epoch=1,
        sizes={0: ResolutionConfig(size=[16, 16], batch_size=BATCH_SIZE)},
    )

    def train_dataloader_fn(size: Sequence[int], batch_size: int) -> DataLoader:
        return make_dataloader(list(size), batch_size)

    def val_dataloader_fn(size: Sequence[int], batch_size: int) -> DataLoader:
        return make_dataloader(list(size), batch_size)

    original_size_change = finetune_module.size_change
    size_change_mock = mocker.patch.object(
        finetune_module,
        "size_change",
        side_effect=lambda *args: original_size_change(*args),
    )
    wandb_log = mocker.patch.object(finetune_module.wandb, "log")
    mocker.patch.object(finetune_module, "format_pbar_description", return_value="desc")

    train(
        model,
        train_dataloader_fn,
        val_dataloader_fn,
        optimizer,
        scheduler,
        trainer_config,
    )

    size_change_mock.assert_called_once()
    assert wandb_log.call_count >= 2
    logged_payloads = [call.args[0] for call in wandb_log.call_args_list]

    train_logs = [payload for payload in logged_payloads if "train/acc" in payload]
    val_logs = [payload for payload in logged_payloads if "val/acc" in payload]

    assert train_logs
    assert val_logs
    assert "train/loss" in train_logs[0]
    assert "train/lr" in train_logs[0]
    assert "train/loss_jepa" not in train_logs[0]
    assert not any("cpa" in key for key in train_logs[0])
    assert "val/acc" in val_logs[0]
