from types import SimpleNamespace
from typing import cast

import torch
from mjepa import CLSPredictionMode, JEPAConfig
from mjepa.jepa import (
    ADALN_BLIND_CLS_PREDICTION_MODE,
    PROJECTED_CLS_PREDICTION_MODE,
    CrossAttentionPredictor,
)
from mjepa.model import MJEPA
from pytest_mock import MockerFixture
from vit import ViTConfig

from mjepa_cifar10.research.cls_path_benchmark import _run_cls_prediction_path, count_cls_prediction_path_parameters


def _predictor(cls_prediction_mode: CLSPredictionMode, num_cls_tokens: int) -> CrossAttentionPredictor:
    backbone = ViTConfig(
        in_channels=3,
        patch_size=[4, 4],
        img_size=[8, 8],
        depth=1,
        hidden_size=16,
        ffn_hidden_size=32,
        num_attention_heads=4,
        num_cls_tokens=num_cls_tokens,
        dtype=torch.float32,
    ).instantiate()
    config = JEPAConfig(cls_prediction_mode=cls_prediction_mode)
    return CrossAttentionPredictor(
        backbone,
        depth=config.predictor_depth,
        attention_mode=config.predictor_attention_mode,
        cls_prediction_mode=config.cls_prediction_mode,
    )


def test_blind_cls_path_parameter_count_excludes_attention_parameters() -> None:
    legacy = _predictor("legacy_cross_attention", num_cls_tokens=4)
    blind = _predictor(ADALN_BLIND_CLS_PREDICTION_MODE, num_cls_tokens=1)

    legacy_count = count_cls_prediction_path_parameters(legacy)
    blind_count = count_cls_prediction_path_parameters(blind)

    assert legacy_count == sum(parameter.numel() for parameter in legacy.parameters())
    assert 0 < blind_count < sum(parameter.numel() for parameter in blind.parameters())
    assert blind_count < legacy_count


def test_projected_cls_path_parameter_count_includes_projection_and_legacy_predictor() -> None:
    legacy = _predictor("legacy_cross_attention", num_cls_tokens=4)
    projected = _predictor(PROJECTED_CLS_PREDICTION_MODE, num_cls_tokens=1)

    projected_count = count_cls_prediction_path_parameters(projected)

    assert projected_count == sum(parameter.numel() for parameter in projected.parameters())
    assert projected_count > sum(parameter.numel() for parameter in legacy.parameters())


def test_projected_cls_benchmark_executes_projection_before_legacy_predictor(mocker: MockerFixture) -> None:
    cls_tokens = torch.randn(2, 1, 16)
    projected_context = torch.randn(2, 4, 16)
    target_mask = torch.zeros(2, 4, dtype=torch.bool)
    output = torch.randn(2, 1, 16)
    predictor = SimpleNamespace(project_cls_context=mocker.Mock(return_value=projected_context))
    jepa = SimpleNamespace(
        config=SimpleNamespace(cls_prediction_mode=PROJECTED_CLS_PREDICTION_MODE),
        predictor=predictor,
        forward_predictor=mocker.Mock(return_value=output),
        forward_blind_cls_predictor=mocker.Mock(),
    )

    result = _run_cls_prediction_path(cast(MJEPA, jepa), (2, 2), cls_tokens, target_mask)

    predictor.project_cls_context.assert_called_once_with(cls_tokens)
    jepa.forward_predictor.assert_called_once_with((2, 2), projected_context, None, target_mask, rope_seed=0)
    assert result is output
