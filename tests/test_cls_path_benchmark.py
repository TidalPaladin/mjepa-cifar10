from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjepa import CLSPredictionMode, JEPAConfig
from mjepa.jepa import (
    ADALN_BLIND_CLS_PREDICTION_MODE,
    JOINT_CONTEXT_CLS_PREDICTION_MODE,
    PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODE,
    PARTITIONED_INDEPENDENT_CLS_PREDICTION_MODE,
    PARTITIONED_SHARED_CLS_PREDICTION_MODE,
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


def _predictor(
    cls_prediction_mode: CLSPredictionMode,
    num_cls_tokens: int,
    cls_context_tokens: int = 4,
) -> CrossAttentionPredictor:
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
    config = JEPAConfig(
        cls_prediction_mode=cls_prediction_mode,
        cls_context_tokens=cls_context_tokens,
    )
    return CrossAttentionPredictor(
        backbone,
        depth=config.predictor_depth,
        attention_mode=config.predictor_attention_mode,
        cls_prediction_mode=config.cls_prediction_mode,
        cls_context_tokens=config.cls_context_tokens,
    )


def test_complete_predictor_workload_parameter_count_includes_all_predictor_parameters() -> None:
    legacy = _predictor("legacy_cross_attention", num_cls_tokens=4)
    blind = _predictor(ADALN_BLIND_CLS_PREDICTION_MODE, num_cls_tokens=1)

    legacy_count = count_cls_prediction_path_parameters(legacy)
    blind_count = count_cls_prediction_path_parameters(blind)

    assert legacy_count == sum(parameter.numel() for parameter in legacy.parameters())
    assert blind_count == sum(parameter.numel() for parameter in blind.parameters())


def test_projected_cls_path_parameter_count_includes_projection_and_legacy_predictor() -> None:
    legacy = _predictor("legacy_cross_attention", num_cls_tokens=4)
    slot_bias = _predictor(SLOT_BIAS_CLS_PREDICTION_MODE, num_cls_tokens=1)
    projected = _predictor(PROJECTED_CLS_PREDICTION_MODE, num_cls_tokens=1)
    residual_projected = _predictor(RESIDUAL_PROJECTED_CLS_PREDICTION_MODE, num_cls_tokens=1)
    residual_mlp = _predictor(RESIDUAL_MLP_CLS_PREDICTION_MODE, num_cls_tokens=1)
    partitioned_shared = _predictor(PARTITIONED_SHARED_CLS_PREDICTION_MODE, num_cls_tokens=1)
    partitioned_independent = _predictor(PARTITIONED_INDEPENDENT_CLS_PREDICTION_MODE, num_cls_tokens=1)

    slot_bias_count = count_cls_prediction_path_parameters(slot_bias)
    projected_count = count_cls_prediction_path_parameters(projected)
    residual_projected_count = count_cls_prediction_path_parameters(residual_projected)
    residual_mlp_count = count_cls_prediction_path_parameters(residual_mlp)
    partitioned_shared_count = count_cls_prediction_path_parameters(partitioned_shared)
    partitioned_independent_count = count_cls_prediction_path_parameters(partitioned_independent)

    assert slot_bias_count == sum(parameter.numel() for parameter in slot_bias.parameters())
    assert projected_count == sum(parameter.numel() for parameter in projected.parameters())
    assert residual_projected_count == sum(parameter.numel() for parameter in residual_projected.parameters())
    assert residual_mlp_count == sum(parameter.numel() for parameter in residual_mlp.parameters())
    assert partitioned_shared_count == sum(parameter.numel() for parameter in partitioned_shared.parameters())
    assert partitioned_independent_count == sum(parameter.numel() for parameter in partitioned_independent.parameters())
    assert sum(parameter.numel() for parameter in legacy.parameters()) < slot_bias_count
    assert (
        slot_bias_count
        < partitioned_shared_count
        < partitioned_independent_count
        < projected_count
        == residual_projected_count
        < residual_mlp_count
    )


@pytest.mark.parametrize("cls_context_tokens", [2, 8])
def test_partitioned_cls_benchmark_supports_configurable_context_count(cls_context_tokens: int) -> None:
    predictor = _predictor(
        PARTITIONED_INDEPENDENT_CLS_PREDICTION_MODE,
        num_cls_tokens=1,
        cls_context_tokens=cls_context_tokens,
    )

    expanded = predictor.expand_cls_context(torch.randn(2, 1, 16))

    assert expanded.shape == (2, cls_context_tokens, 16)


def test_cls_benchmark_executes_both_legacy_predictor_forwards(mocker: MockerFixture) -> None:
    visual_context = torch.randn(2, 2, 16)
    cls_tokens = torch.randn(2, 1, 16)
    context_mask = torch.zeros(2, 4, dtype=torch.bool)
    target_mask = torch.zeros(2, 4, dtype=torch.bool)
    visual_output = torch.randn(2, 1, 16)
    cls_output = torch.randn(2, 1, 16)
    jepa = mocker.Mock(spec=MJEPA)
    jepa.config = SimpleNamespace(cls_prediction_mode="legacy_cross_attention")
    jepa.forward_predictor.return_value = visual_output
    jepa.forward_cls_predictor.return_value = cls_output

    result = _run_cls_prediction_path(
        cast(MJEPA, jepa),
        (2, 2),
        visual_context,
        cls_tokens,
        context_mask,
        target_mask,
    )

    jepa.forward_predictor.assert_called_once_with(
        (2, 2),
        visual_context,
        context_mask,
        target_mask,
        rope_seed=0,
    )
    jepa.forward_cls_predictor.assert_called_once_with((2, 2), cls_tokens, target_mask, rope_seed=0)
    assert result == (visual_output, cls_output)


def test_cls_benchmark_executes_one_joint_context_predictor_forward(mocker: MockerFixture) -> None:
    visual_context = torch.randn(2, 2, 16)
    cls_tokens = torch.randn(2, 1, 16)
    context_mask = torch.zeros(2, 4, dtype=torch.bool)
    target_mask = torch.zeros(2, 4, dtype=torch.bool)
    output = torch.randn(2, 1, 16)
    jepa = mocker.Mock(spec=MJEPA)
    jepa.config = SimpleNamespace(cls_prediction_mode=JOINT_CONTEXT_CLS_PREDICTION_MODE)
    jepa.forward_joint_context_predictor_heads.return_value = (output, None)

    result = _run_cls_prediction_path(
        cast(MJEPA, jepa),
        (2, 2),
        visual_context,
        cls_tokens,
        context_mask,
        target_mask,
    )

    jepa.forward_joint_context_predictor_heads.assert_called_once_with(
        (2, 2),
        visual_context,
        cls_tokens,
        context_mask,
        target_mask,
        rope_seed=0,
    )
    jepa.forward_predictor.assert_not_called()
    jepa.forward_cls_predictor.assert_not_called()
    assert result == (output, None)


def test_cls_benchmark_executes_one_packed_adaln_predictor_forward(mocker: MockerFixture) -> None:
    visual_context = torch.randn(2, 2, 16)
    cls_tokens = torch.randn(2, 1, 16)
    context_mask = torch.zeros(2, 4, dtype=torch.bool)
    target_mask = torch.zeros(2, 4, dtype=torch.bool)
    output = torch.randn(2, 2, 16)
    jepa = mocker.Mock(spec=MJEPA)
    jepa.config = SimpleNamespace(cls_prediction_mode=PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODE)
    jepa.forward_packed_adaln_hard_blind_predictor_heads.return_value = (output, None)

    result = _run_cls_prediction_path(
        cast(MJEPA, jepa),
        (2, 2),
        visual_context,
        cls_tokens,
        context_mask,
        target_mask,
    )

    jepa.forward_packed_adaln_hard_blind_predictor_heads.assert_called_once_with(
        (2, 2),
        visual_context,
        cls_tokens,
        context_mask,
        target_mask,
        rope_seed=0,
    )
    jepa.forward_predictor.assert_not_called()
    jepa.forward_cls_predictor.assert_not_called()
    assert result == (output, None)
