import math
from pathlib import Path

import pytest
import torch
from vit import ViT, ViTConfig

from mjepa_cifar10.probe_calibration import (
    FINAL_CLS_MODE,
    LAST_TWO_CLS_MODE,
    FrozenBackboneProbe,
    LinearProbeBank,
    extract_probe_features,
    load_feature_cache,
    save_feature_cache,
    train_probe_bank,
    warmup_cosine_factor,
)


HIDDEN_SIZE = 8
NUM_CLASSES = 3
BATCH_SIZE = 4


def make_backbone() -> ViT:
    config = ViTConfig(
        in_channels=3,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=16,
        patch_size=[4, 4],
        img_size=[8, 8],
        depth=2,
        num_attention_heads=2,
        hidden_dropout=0.2,
        attention_dropout=0.2,
        num_register_tokens=2,
        num_cls_tokens=2,
        dtype=torch.float32,
    )
    return config.instantiate()


def manually_extract_cls_layers(backbone: ViT, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized_size = backbone.stem.tokenized_size(images.shape[2:])
    dense_features = backbone.normalize_patch_embeddings(backbone.stem(images))
    dense_features = backbone.add_prefix_tokens(dense_features)
    rope = backbone.prepare_rope(tokenized_size)
    layer_cls_features = []
    for block in backbone.blocks:
        dense_features = block(dense_features, rope=rope)
        normalized_features = backbone.output_norm(dense_features)
        layer_cls_features.append(normalized_features[:, : backbone.config.num_cls_tokens].mean(dim=1))
    return layer_cls_features[-2], layer_cls_features[-1]


def test_extract_probe_features_matches_normalized_final_two_cls_layers() -> None:
    torch.manual_seed(0)
    backbone = make_backbone()
    backbone.eval()
    images = torch.randn(BATCH_SIZE, 3, 8, 8)

    with torch.no_grad():
        expected_penultimate, expected_final = manually_extract_cls_layers(backbone, images)
        extracted = extract_probe_features(backbone, images)

    assert torch.allclose(extracted[FINAL_CLS_MODE], expected_final)
    assert torch.allclose(
        extracted[LAST_TWO_CLS_MODE],
        torch.cat((expected_penultimate, expected_final), dim=-1),
    )


def test_frozen_backbone_probe_keeps_encoder_in_eval_and_detached() -> None:
    backbone = make_backbone()
    probe = FrozenBackboneProbe(
        backbone,
        mode=LAST_TWO_CLS_MODE,
        num_classes=NUM_CLASSES,
        normalize=True,
    )
    images = torch.randn(BATCH_SIZE, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 0])

    probe.train()
    logits = probe(images)
    torch.nn.functional.cross_entropy(logits, labels).backward()

    assert probe.training
    assert not probe.backbone.training
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert all(not parameter.requires_grad for parameter in probe.backbone.parameters())
    assert all(parameter.grad is None for parameter in probe.backbone.parameters())
    assert all(parameter.grad is not None for parameter in probe.classifier.parameters())
    assert not tuple(probe.normalization.parameters())


def test_frozen_backbone_probe_rejects_last_two_mode_for_one_layer_encoder() -> None:
    config = ViTConfig(
        in_channels=3,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=16,
        patch_size=[4, 4],
        img_size=[8, 8],
        depth=1,
        num_attention_heads=2,
        num_cls_tokens=1,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="at least two encoder layers"):
        FrozenBackboneProbe(
            config.instantiate(),
            mode=LAST_TWO_CLS_MODE,
            num_classes=NUM_CLASSES,
            normalize=True,
        )


@pytest.mark.parametrize(
    ("step", "expected"),
    (
        (0, 0.1),
        (5, 1.0),
        (10, 0.001),
    ),
)
def test_warmup_cosine_factor_hits_preregistered_endpoints(step: int, expected: float) -> None:
    factor = warmup_cosine_factor(
        step,
        total_steps=10,
        warmup_steps=5,
        start_factor=0.1,
        final_factor=0.001,
    )

    assert factor == pytest.approx(expected)


def test_warmup_cosine_factor_follows_cosine_after_warmup() -> None:
    factor = warmup_cosine_factor(
        7,
        total_steps=10,
        warmup_steps=5,
        start_factor=0.1,
        final_factor=0.001,
    )
    progress = 2 / 5
    expected = 0.001 + 0.5 * (1.0 - 0.001) * (1.0 + math.cos(math.pi * progress))

    assert factor == pytest.approx(expected)


def test_linear_probe_bank_starts_each_learning_rate_from_identical_weights() -> None:
    torch.manual_seed(0)
    bank = LinearProbeBank(feature_size=HIDDEN_SIZE, num_classes=NUM_CLASSES, learning_rates=(1e-3, 1e-2))
    features = torch.randn(BATCH_SIZE, HIDDEN_SIZE)

    logits = bank(features)

    assert logits.shape == (2, BATCH_SIZE, NUM_CLASSES)
    assert torch.equal(logits[0], logits[1])
    assert bank.classifiers[0].weight is not bank.classifiers[1].weight


def test_feature_cache_round_trip_is_atomic_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "features.safetensors"
    tensors = {
        FINAL_CLS_MODE: torch.randn(BATCH_SIZE, HIDDEN_SIZE),
        LAST_TWO_CLS_MODE: torch.randn(BATCH_SIZE, HIDDEN_SIZE * 2),
        "labels": torch.tensor([0, 1, 2, 0]),
    }

    save_feature_cache(path, tensors)
    loaded = load_feature_cache(path)

    assert set(loaded) == set(tensors)
    assert all(torch.equal(loaded[key], value) for key, value in tensors.items())
    assert not tuple(tmp_path.glob("*.tmp"))


def test_train_probe_bank_fits_linearly_separable_cached_features() -> None:
    torch.manual_seed(0)
    examples_per_class = 12
    labels = torch.arange(NUM_CLASSES).repeat_interleave(examples_per_class)
    features = torch.nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()
    features = torch.cat((features, torch.zeros(len(labels), HIDDEN_SIZE - NUM_CLASSES)), dim=1)
    train_features = features + torch.randn_like(features) * 0.01
    val_features = features + torch.randn_like(features) * 0.01

    result = train_probe_bank(
        train_features,
        labels,
        val_features,
        labels,
        learning_rates=(0.01, 0.1),
        epochs=20,
        batch_size=12,
        weight_decay=1e-6,
        warmup_fraction=0.1,
        start_factor=0.1,
        final_factor=0.001,
        device=torch.device("cpu"),
        seed=0,
    )

    assert result.best_peak_accuracy > 0.95
    assert result.best_learning_rate in (0.01, 0.1)
    assert len(result.validation_curves) == 2
    assert all(len(curve) == 20 for curve in result.validation_curves)
