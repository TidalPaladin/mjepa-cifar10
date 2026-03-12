from types import SimpleNamespace

import pytest
import torch
from mjepa import JEPAConfig
from mjepa.jepa import CrossAttentionPredictor
from mjepa.metrics import CLSPatchAlignmentMetric
from torch import Tensor
from torch import nn
from vit import AttentivePoolHeadConfig
from vit import ViTConfig
from vit import ViTFeatures

from mjepa_cifar10.pretrain import CPA_RESULT_KEYS
from mjepa_cifar10.pretrain import CIFAR10MJEPA
from mjepa_cifar10.pretrain import compute_and_reset_cpa_metrics
from mjepa_cifar10.pretrain import get_scheduler_last_lr
from mjepa_cifar10.pretrain import update_cls_patch_alignment_metric


def test_get_scheduler_last_lr_returns_first_learning_rate() -> None:
    scheduler = SimpleNamespace(get_last_lr=lambda: [0.2, 0.1])

    assert get_scheduler_last_lr(scheduler) == 0.2


HIDDEN_SIZE = 8
NUM_REGISTER_TOKENS = 2
NUM_CLS_TOKENS = 2
NUM_VISUAL_TOKENS = 4
BATCH_SIZE = 2


class RecordingHead(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.last_input: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.clone()
        if x.ndim != 2:
            raise AssertionError(f"expected a pooled embedding, got shape={tuple(x.shape)}")
        return x[:, : self.out_features]


def make_model(
    *,
    num_cls_tokens: int,
    head_config: AttentivePoolHeadConfig | None = None,
) -> CIFAR10MJEPA:
    backbone_config = ViTConfig(
        in_channels=3,
        hidden_size=HIDDEN_SIZE,
        patch_size=[4, 4],
        img_size=[8, 8],
        depth=1,
        num_attention_heads=2,
        ffn_hidden_size=16,
        num_register_tokens=NUM_REGISTER_TOKENS,
        num_cls_tokens=num_cls_tokens,
        dtype=torch.float32,
        heads={"cls": head_config} if head_config is not None else {},
    )
    backbone = backbone_config.instantiate()
    predictor = CrossAttentionPredictor(backbone, depth=1)
    return CIFAR10MJEPA(JEPAConfig(gram_start_epoch=None), backbone, predictor, autocast_dtype=torch.float32)


def make_features(*, num_cls_tokens: int) -> ViTFeatures:
    cls_count = num_cls_tokens
    total_tokens = cls_count + NUM_REGISTER_TOKENS + NUM_VISUAL_TOKENS
    dense_features = torch.arange(BATCH_SIZE * total_tokens * HIDDEN_SIZE, dtype=torch.float32).view(
        BATCH_SIZE, total_tokens, HIDDEN_SIZE
    )
    return ViTFeatures(dense_features, NUM_REGISTER_TOKENS, cls_count, tokenized_size=(2, 2))


def test_forward_probe_pools_cls_tokens_before_linear_head(mocker) -> None:
    model = make_model(num_cls_tokens=NUM_CLS_TOKENS)
    features = make_features(num_cls_tokens=NUM_CLS_TOKENS)
    head = RecordingHead(out_features=3)
    mocker.patch.object(model.student, "get_head", return_value=head)

    output = model.forward_probe(features)

    assert head.last_input is not None
    assert torch.equal(head.last_input, features.cls_tokens.mean(1))
    assert output["cls"].shape == (BATCH_SIZE, 3)


def test_forward_probe_uses_attentive_pooling_for_visual_tokens_without_cls() -> None:
    model = make_model(
        num_cls_tokens=0,
        head_config=AttentivePoolHeadConfig(
            out_features=3,
            num_attention_heads=2,
            num_queries=1,
        ),
    )
    features = make_features(num_cls_tokens=0)

    output = model.forward_probe(features)

    assert output["cls"].shape == (BATCH_SIZE, 3)
    assert torch.isfinite(output["cls"]).all()


def test_forward_probe_requires_single_embedding_when_cls_tokens_are_disabled(mocker) -> None:
    model = make_model(num_cls_tokens=0)
    features = make_features(num_cls_tokens=0)
    mocker.patch.object(model.student, "get_head", return_value=nn.Identity())

    with pytest.raises(ValueError, match="single embedding per sample"):
        model.forward_probe(features)


def test_update_cls_patch_alignment_metric_updates_metric_from_features() -> None:
    metric = CLSPatchAlignmentMetric(num_bins=4096)
    features = make_features(num_cls_tokens=NUM_CLS_TOKENS)

    assert update_cls_patch_alignment_metric(metric, features) is True

    out = metric.compute()
    cls_norm = torch.nn.functional.normalize(features.cls_tokens, dim=-1)
    patch_norm = torch.nn.functional.normalize(features.visual_tokens, dim=-1)
    expected = torch.einsum("bcd,bnd->bcn", cls_norm, patch_norm).reshape(-1)
    assert torch.allclose(out["cpa_mean"], expected.mean().to(out["cpa_mean"].dtype), atol=1e-7)
    assert torch.allclose(out["cpa_std"], expected.std(unbiased=False).to(out["cpa_std"].dtype), atol=1e-7)


def test_update_cls_patch_alignment_metric_skips_features_without_cls_tokens() -> None:
    metric = CLSPatchAlignmentMetric()
    features = make_features(num_cls_tokens=0)

    assert update_cls_patch_alignment_metric(metric, features) is False
    assert metric.count.item() == 0
    assert metric.hist.sum().item() == 0.0


def test_compute_and_reset_cpa_metrics_prefixes_keys_and_resets_state() -> None:
    metric = CLSPatchAlignmentMetric(num_bins=4096)
    metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]))
    expected_metrics = {key: value.item() for key, value in metric.compute().items()}

    logged_metrics = compute_and_reset_cpa_metrics(metric, prefix="train")

    assert logged_metrics == {f"train/{key}": pytest.approx(value) for key, value in expected_metrics.items()}
    assert tuple(key.removeprefix("train/") for key in logged_metrics) == CPA_RESULT_KEYS
