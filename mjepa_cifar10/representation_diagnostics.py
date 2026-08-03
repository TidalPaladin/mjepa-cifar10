from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor
from vit import ViT, ViTFeatures


CLS_ROUTE: Final[str] = "cls"
PATCH_MEAN_ROUTE: Final[str] = "patch_mean"
CLS_PATCH_MEAN_ROUTE: Final[str] = "cls_patch_mean"
REPRESENTATION_ROUTES: Final[tuple[str, ...]] = (CLS_ROUTE, PATCH_MEAN_ROUTE, CLS_PATCH_MEAN_ROUTE)


def extract_normalized_layer_features(backbone: ViT, images: Tensor) -> tuple[ViTFeatures, ...]:
    """Return every encoder block output after applying the backbone's final normalization."""
    tokenized_size = backbone.stem.tokenized_size(images.shape[2:])
    dense_features = backbone.normalize_patch_embeddings(backbone.stem(images))
    dense_features = backbone.add_prefix_tokens(dense_features)
    rope = backbone.prepare_rope(tokenized_size)
    layer_features: list[ViTFeatures] = []
    for block in backbone.blocks:
        dense_features = block(dense_features, rope=rope)
        normalized = backbone.output_norm(dense_features)
        layer_features.append(
            ViTFeatures(
                normalized,
                backbone.config.num_register_tokens,
                backbone.config.num_cls_tokens,
                tokenized_size,
            )
        )
    return tuple(layer_features)


def representation_routes(features: ViTFeatures) -> dict[str, Tensor]:
    """Construct global diagnostic routes without mixing register tokens into image features."""
    if features.num_cls_tokens <= 0:
        raise ValueError("representation diagnostics require at least one CLS token")
    if features.visual_tokens.shape[1] <= 0:
        raise ValueError("representation diagnostics require at least one visual token")
    cls = features.cls_tokens.mean(dim=1)
    patch_mean = features.visual_tokens.mean(dim=1)
    return {
        CLS_ROUTE: cls,
        PATCH_MEAN_ROUTE: patch_mean,
        CLS_PATCH_MEAN_ROUTE: torch.cat((cls, patch_mean), dim=-1),
    }


def _normalize_probe_features(features: Tensor) -> Tensor:
    normalized = F.layer_norm(features.float(), (features.shape[-1],))
    return F.normalize(normalized, dim=-1)


class NearestCentroidProbe:
    """Streaming deterministic cosine-centroid probe for representation diagnosis."""

    def __init__(self, *, num_classes: int, feature_dim: int, device: torch.device) -> None:
        if num_classes <= 1 or feature_dim <= 0:
            raise ValueError("num_classes must exceed one and feature_dim must be positive")
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_sums = torch.zeros(num_classes, feature_dim, dtype=torch.float64, device=device)
        self.class_counts = torch.zeros(num_classes, dtype=torch.int64, device=device)
        self.prototypes: Tensor | None = None
        self.correct = torch.zeros((), dtype=torch.int64, device=device)
        self.total = torch.zeros((), dtype=torch.int64, device=device)

    def update_train(self, features: Tensor, labels: Tensor) -> None:
        if self.prototypes is not None:
            raise RuntimeError("training centroids are already finalized")
        if features.ndim != 2 or features.shape[1] != self.feature_dim or len(features) != len(labels):
            raise ValueError("training features and labels have incompatible shapes")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError("training labels fall outside the configured class range")
        normalized = _normalize_probe_features(features).to(dtype=torch.float64, device=self.class_sums.device)
        labels = labels.to(device=self.class_counts.device, dtype=torch.long)
        self.class_sums.index_add_(0, labels, normalized)
        self.class_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.int64))

    def finalize(self) -> None:
        if (self.class_counts == 0).any():
            raise ValueError("nearest-centroid training requires at least one example from every class")
        centroids = self.class_sums / self.class_counts.unsqueeze(-1)
        self.prototypes = F.normalize(centroids.float(), dim=-1)

    def update_validation(self, features: Tensor, labels: Tensor) -> None:
        if self.prototypes is None:
            raise RuntimeError("training centroids must be finalized before validation")
        if features.ndim != 2 or features.shape[1] != self.feature_dim or len(features) != len(labels):
            raise ValueError("validation features and labels have incompatible shapes")
        normalized = _normalize_probe_features(features).to(device=self.prototypes.device)
        labels = labels.to(device=self.prototypes.device, dtype=torch.long)
        predictions = (normalized @ self.prototypes.T).argmax(dim=-1)
        self.correct += (predictions == labels).sum()
        self.total += len(labels)

    def compute_accuracy(self) -> float:
        if self.total == 0:
            raise ValueError("nearest-centroid validation received no examples")
        return float((self.correct / self.total).item())
