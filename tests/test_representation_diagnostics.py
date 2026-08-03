import torch
from vit import ViT, ViTConfig

from mjepa_cifar10.representation_diagnostics import (
    NearestCentroidProbe,
    extract_normalized_layer_features,
    representation_routes,
)


HIDDEN_SIZE = 8
NUM_CLASSES = 2
BATCH_SIZE = 4


def make_backbone() -> ViT:
    return ViTConfig(
        in_channels=3,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=16,
        patch_size=[4, 4],
        img_size=[8, 8],
        depth=2,
        num_attention_heads=2,
        num_register_tokens=2,
        num_cls_tokens=2,
        dtype=torch.float32,
    ).instantiate()


def test_extract_normalized_layer_features_matches_backbone_final_output() -> None:
    torch.manual_seed(0)
    backbone = make_backbone().eval()
    images = torch.randn(BATCH_SIZE, 3, 8, 8)

    with torch.inference_mode():
        layers = extract_normalized_layer_features(backbone, images)
        expected = backbone(images)

    assert len(layers) == len(backbone.blocks)
    assert torch.allclose(layers[-1].dense_features, expected.dense_features)


def test_representation_routes_pool_cls_and_patch_tokens_separately() -> None:
    backbone = make_backbone().eval()
    images = torch.randn(BATCH_SIZE, 3, 8, 8)

    with torch.inference_mode():
        features = extract_normalized_layer_features(backbone, images)[-1]
        routes = representation_routes(features)

    assert routes["cls"].shape == (BATCH_SIZE, HIDDEN_SIZE)
    assert routes["patch_mean"].shape == (BATCH_SIZE, HIDDEN_SIZE)
    assert routes["cls_patch_mean"].shape == (BATCH_SIZE, HIDDEN_SIZE * 2)
    assert torch.allclose(routes["cls"], features.cls_tokens.mean(dim=1))
    assert torch.allclose(routes["patch_mean"], features.visual_tokens.mean(dim=1))


def test_nearest_centroid_probe_classifies_separated_clusters() -> None:
    probe = NearestCentroidProbe(num_classes=NUM_CLASSES, feature_dim=2, device=torch.device("cpu"))
    train_features = torch.tensor(((2.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.0, 1.0)))
    train_labels = torch.tensor((0, 0, 1, 1))
    validation_features = torch.tensor(((3.0, 0.0), (0.0, 3.0), (1.0, 0.1), (0.1, 1.0)))
    validation_labels = torch.tensor((0, 1, 0, 1))

    probe.update_train(train_features, train_labels)
    probe.finalize()
    probe.update_validation(validation_features, validation_labels)

    assert probe.compute_accuracy() == 1.0


def test_nearest_centroid_probe_requires_every_class() -> None:
    probe = NearestCentroidProbe(num_classes=NUM_CLASSES, feature_dim=2, device=torch.device("cpu"))
    probe.update_train(torch.tensor(((1.0, 0.0),)), torch.tensor((0,)))

    try:
        probe.finalize()
    except ValueError as error:
        assert "every class" in str(error)
    else:
        raise AssertionError("missing classes must be rejected")
