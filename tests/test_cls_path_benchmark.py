from typing import cast

import torch
from mjepa import CLSPredictionMode, JEPAConfig
from mjepa.jepa import (
    ADALN_BLIND_CLS_PREDICTION_MODE,
    PROJECTED_CLS_PREDICTION_MODE,
    RESIDUAL_MLP_CLS_PREDICTION_MODE,
    RESIDUAL_PROJECTED_CLS_PREDICTION_MODE,
    SLOT_BIAS_CLS_PREDICTION_MODE,
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
    slot_bias = _predictor(SLOT_BIAS_CLS_PREDICTION_MODE, num_cls_tokens=1)
    projected = _predictor(PROJECTED_CLS_PREDICTION_MODE, num_cls_tokens=1)
    residual_projected = _predictor(RESIDUAL_PROJECTED_CLS_PREDICTION_MODE, num_cls_tokens=1)
    residual_mlp = _predictor(RESIDUAL_MLP_CLS_PREDICTION_MODE, num_cls_tokens=1)

    slot_bias_count = count_cls_prediction_path_parameters(slot_bias)
    projected_count = count_cls_prediction_path_parameters(projected)
    residual_projected_count = count_cls_prediction_path_parameters(residual_projected)
    residual_mlp_count = count_cls_prediction_path_parameters(residual_mlp)

    assert slot_bias_count == sum(parameter.numel() for parameter in slot_bias.parameters())
    assert projected_count == sum(parameter.numel() for parameter in projected.parameters())
    assert residual_projected_count == sum(parameter.numel() for parameter in residual_projected.parameters())
    assert residual_mlp_count == sum(parameter.numel() for parameter in residual_mlp.parameters())
    assert sum(parameter.numel() for parameter in legacy.parameters()) < slot_bias_count
    assert slot_bias_count < projected_count == residual_projected_count < residual_mlp_count


def test_cls_benchmark_executes_the_configured_auxiliary_path(mocker: MockerFixture) -> None:
    cls_tokens = torch.randn(2, 1, 16)
    target_mask = torch.zeros(2, 4, dtype=torch.bool)
    output = torch.randn(2, 1, 16)
    jepa = mocker.Mock(spec=MJEPA)
    jepa.forward_cls_predictor.return_value = output

    result = _run_cls_prediction_path(cast(MJEPA, jepa), (2, 2), cls_tokens, target_mask)

    jepa.forward_cls_predictor.assert_called_once_with((2, 2), cls_tokens, target_mask, rope_seed=0)
    assert result is output
