import torch
from mjepa import CLSPredictionMode, JEPAConfig
from mjepa.jepa import ADALN_BLIND_CLS_PREDICTION_MODE, CrossAttentionPredictor
from vit import ViTConfig

from mjepa_cifar10.research.cls_path_benchmark import count_cls_prediction_path_parameters


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
