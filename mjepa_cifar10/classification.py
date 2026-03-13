from typing import Final

from torch import Tensor
from vit import ViT, ViTFeatures


CLASSIFIER_HEAD_NAME: Final[str] = "cls"


def flatten_classifier_logits(logits: Tensor) -> Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3 and logits.shape[1] == 1:
        return logits[:, 0, :]
    raise ValueError(f"probe head must return a single embedding per sample, got shape={tuple(logits.shape)}")


def forward_classifier(backbone: ViT, features: ViTFeatures, head_name: str = CLASSIFIER_HEAD_NAME) -> Tensor:
    probe_tokens = features.cls_tokens if features.num_cls_tokens > 0 else features.visual_tokens
    probe_input = probe_tokens.mean(1) if features.num_cls_tokens > 0 else probe_tokens
    logits = backbone.get_head(head_name)(probe_input)
    return flatten_classifier_logits(logits)
