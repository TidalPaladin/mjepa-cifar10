import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from mjepa import JEPAConfig, OptimizerConfig
from mjepa.trainer import CheckpointMetadata
from vit import ViTConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAIN_CONFIG_PATH = REPO_ROOT / "config" / "pretrain" / "vit-small.yaml"
DEFAULT_FINETUNE_CONFIG_PATH = REPO_ROOT / "config" / "finetune" / "vit-small.yaml"
LEGACY_PRETRAIN_CONFIG_PATH = REPO_ROOT / "config" / "pretrain" / "vit-small-four-cls-legacy.yaml"
LEGACY_FINETUNE_CONFIG_PATH = REPO_ROOT / "config" / "finetune" / "vit-small-four-cls-legacy.yaml"
LEGACY_SINGLE_CLS_FINETUNE_CONFIG_PATH = REPO_ROOT / "config" / "finetune" / "vit-small-single-cls-four-registers.yaml"
SELECTED_NUM_CLS_TOKENS = 1
SELECTED_NUM_REGISTER_TOKENS = 7
SELECTED_CLS_CONTEXT_TOKENS = 4
SELECTED_CLS_PREDICTION_MODE = "partitioned_independent_cross_attention"
MULTIVIEW_EFFECTIVE_BATCH_SIZE = 1024
TOKEN_SPECIALIZATION_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-ijepa-token-specialization.yaml": (
        DEFAULT_PRETRAIN_CONFIG_PATH,
        4,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-ijepa-token-specialization-smoke.yaml": (
        REPO_ROOT / "config" / "pretrain" / "smoke.yaml",
        1,
    ),
}
HISTORICAL_FOUR_CLS_STUDY_IDS = (
    "cls-global-target-v1",
    "cls-partition-count-v1",
    "cls-partitioned-slots-v1",
    "cls-register-residual-v1",
    "cls-register-slots-v1",
    "cls-token-adaln-v1",
    "cls-up-project-v1",
    "muon-optimizer-v1",
    "muon-optimizer-v2",
    "srelu-mlp-baseline-v1",
    "srelu-mlp-bias-v1",
    "srelu-mlp-width-v1",
    "vit-small-baseline-v1",
)
PRETRAIN_CONFIG_PATHS = (
    DEFAULT_PRETRAIN_CONFIG_PATH,
    REPO_ROOT / "config" / "pretrain" / "vit-tiny.yaml",
)
SRELU_WIDTH_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-srelu-h1536.yaml": 1536,
    REPO_ROOT / "config" / "pretrain" / "vit-small-srelu-h2304.yaml": 2304,
    REPO_ROOT / "config" / "pretrain" / "vit-small-srelu-h2305.yaml": 2305,
}
SRELU_CHANGED_BACKBONE_FIELDS = frozenset(("activation", "ffn_hidden_size"))
SRELU_BIAS_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-srelu-h1536-bias0p1.yaml": 0.1,
    REPO_ROOT / "config" / "pretrain" / "vit-small-srelu-h1536-bias0p2.yaml": 0.2,
}
LEJEPA_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-shared-nosigreg-100e.yaml": (
        None,
        "context",
        (),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-direct-both-l005-100e.yaml": (
        0.05,
        "both",
        (),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-proj64-both-l005-100e.yaml": (
        0.05,
        "both",
        (2048, 2048, 64),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-proj64-target-l005-100e.yaml": (
        0.05,
        "target",
        (2048, 2048, 64),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-smoke.yaml": (
        0.05,
        "both",
        (2048, 2048, 64),
    ),
}
LEJEPA_MASKED_OPTIMIZATION_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-clspatch-smoke.yaml": (
        0.20,
        False,
        True,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-clspatch-both-l005-aux-100e.yaml": (
        0.05,
        True,
        False,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-clspatch-both-l005-noaux-100e.yaml": (
        0.05,
        False,
        False,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-clspatch-both-l020-noaux-100e.yaml": (
        0.20,
        False,
        False,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-clspatch-both-l020-noaux-deterministic-100e.yaml": (
        0.20,
        False,
        True,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-clspatch-both-l010-aux-deterministic-100e.yaml": (
        0.10,
        True,
        True,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-clspatch-both-l020-aux-deterministic-100e.yaml": (
        0.20,
        True,
        True,
    ),
}
LEJEPA_MULTIVIEW_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-multiview-g2-direct-l010-w1-100e.yaml": (
        2,
        0,
        0.10,
        1.0,
        (),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-multiview-g4-direct-l010-w1-100e.yaml": (
        4,
        0,
        0.10,
        1.0,
        (),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-multiview-g2l2-direct-l010-w1-100e.yaml": (
        2,
        2,
        0.10,
        1.0,
        (),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-multiview-g2l2-proj64-l010-w1-100e.yaml": (
        2,
        2,
        0.10,
        1.0,
        (2048, 2048, 64),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-multiview-g2l2-direct-l005-w1-100e.yaml": (
        2,
        2,
        0.05,
        1.0,
        (),
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-multiview-g2l2-direct-l010-w2-100e.yaml": (
        2,
        2,
        0.10,
        2.0,
        (),
    ),
}
LEJEPA_CONVERGENCE_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-convergence-lr2e3-wd2e1-constant-100e.yaml": (
        0.002,
        0.2,
        False,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-convergence-lr5e4-wd2e1-constant-100e.yaml": (
        0.0005,
        0.2,
        False,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-convergence-lr2e3-wd5e2-constant-100e.yaml": (
        0.002,
        0.05,
        False,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-convergence-lr5e4-wd5e2-onecycle-100e.yaml": (
        0.0005,
        0.05,
        True,
    ),
}
LEJEPA_LOSS_VIEW_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-convergence-lr2e3-wd2e1-constant-100e.yaml": (
        2,
        2,
        0.10,
        2.0,
        True,
        [0.30, 0.75],
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-lossview-g2l2-l005-w2-100e.yaml": (
        2,
        2,
        0.05,
        2.0,
        True,
        [0.30, 0.75],
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-lossview-g2l2-l010-w4-100e.yaml": (
        2,
        2,
        0.10,
        4.0,
        True,
        [0.30, 0.75],
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-lossview-g2l2-l005-w4-100e.yaml": (
        2,
        2,
        0.05,
        4.0,
        True,
        [0.30, 0.75],
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-lossview-g2l2-l005-w2-noaux-100e.yaml": (
        2,
        2,
        0.05,
        2.0,
        False,
        [0.30, 0.75],
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-lossview-g2l1-l005-w2-100e.yaml": (
        2,
        1,
        0.05,
        2.0,
        True,
        [0.30, 0.75],
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-lossview-g2-l005-w2-100e.yaml": (
        2,
        0,
        0.05,
        2.0,
        True,
        [0.30, 0.75],
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-lossview-g2l2-l005-w2-local50-100e.yaml": (
        2,
        2,
        0.05,
        2.0,
        True,
        [0.50, 0.75],
    ),
}
LEJEPA_TOKEN_DIVERSITY_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-control-40e.yaml": (
        "cls_patch_mean",
        0.0,
        1.0,
        40,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-cls-only-40e.yaml": (
        "cls",
        0.0,
        1.0,
        40,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-patch-residual-40e.yaml": (
        "cls_patch_mean",
        0.05,
        1.0,
        40,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-mim025-40e.yaml": (
        "cls_patch_mean",
        0.0,
        0.25,
        40,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-mechanical-smoke.yaml": (
        "cls_patch_mean",
        0.05,
        0.25,
        1,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-logging-smoke.yaml": (
        "cls_patch_mean",
        0.05,
        0.25,
        2,
    ),
}
LEJEPA_PATCH_RANK_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-patch-residual-40e.yaml": (0.05, 0.0, 40),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-patch-rank-l010-40e.yaml": (0.10, 0.0, 40),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-patch-rank-sample010-40e.yaml": (0.05, 0.01, 40),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-patch-rank-sample050-40e.yaml": (0.05, 0.05, 40),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-patch-rank-sample050-smoke.yaml": (0.05, 0.05, 1),
}
LEJEPA_TARGET_STOPGRAD_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-target-stopgrad-40e.yaml": 40,
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-target-stopgrad-smoke.yaml": 1,
}
LEJEPA_SINGLE_VIEW_STOPGRAD_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-shared-40e.yaml": (False, 40),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-stopgrad-40e.yaml": (True, 40),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-stopgrad-smoke.yaml": (True, 1),
}
LEJEPA_SINGLE_VIEW_BOTTLENECK_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-stopgrad-lambda010-40e.yaml": (
        "lejepa_lambda",
        0.10,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-stopgrad-patch-residual010-40e.yaml": (
        "patch_residual_sigreg_loss_weight",
        0.10,
    ),
}
CLS_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-legacy.yaml": "legacy_cross_attention",
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind.yaml": "adaln_blind",
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-shared.yaml": "adaln_shared",
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-projected.yaml": "projected_cross_attention",
}
CLS_REGISTER_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-legacy.yaml": "legacy_cross_attention",
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-slot-bias.yaml": "slot_bias_cross_attention",
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-projected.yaml": "projected_cross_attention",
    REPO_ROOT
    / "config"
    / "pretrain"
    / "vit-small-single-cls-register-residual-projected.yaml": "residual_projected_cross_attention",
    REPO_ROOT
    / "config"
    / "pretrain"
    / "vit-small-single-cls-register-residual-mlp.yaml": "residual_mlp_cross_attention",
    REPO_ROOT
    / "config"
    / "pretrain"
    / "vit-small-single-cls-register-partitioned-shared.yaml": "partitioned_shared_cross_attention",
    REPO_ROOT
    / "config"
    / "pretrain"
    / "vit-small-single-cls-register-partitioned-independent.yaml": "partitioned_independent_cross_attention",
}
CLS_PARTITION_COUNT_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-partitioned-independent-2.yaml": 2,
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-partitioned-independent-8.yaml": 8,
}
CLS_JOINT_CONTEXT_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-joint-context-unmasked.yaml": (
        "joint_context",
        2.0,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-joint-context-sample-routed.yaml": (
        "joint_context_sample_routed",
        2.0,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-joint-context-token-routed.yaml": (
        "joint_context_token_routed",
        2.0,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-joint-context-dual-routed.yaml": (
        "joint_context_dual_routed",
        1.0,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-joint-context-packed-dual-routed.yaml": (
        "joint_context_packed_dual_routed",
        1.0,
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-joint-context-token-routed-source-balanced.yaml": (
        "joint_context_token_routed_source_balanced",
        2.0,
    ),
}
CLS_PACKED_ADALN_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind.yaml": (
        "packed_adaln_hard_blind"
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind-adapter.yaml": (
        "packed_adaln_hard_blind_adapter"
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind-mixer.yaml": (
        "packed_adaln_hard_blind_mixer"
    ),
}
CLS_GLOBAL_TARGET_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind-global-w0p1.yaml": 0.1,
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind-global-w0p5.yaml": 0.5,
}
CLS_TEACHER_GLOBAL_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind-global-mean.yaml": (
        "centered_normalized_mean"
    ),
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind-global-ema-attention.yaml": (
        "centered_normalized_ema_attention"
    ),
    REPO_ROOT
    / "config"
    / "pretrain"
    / "vit-small-single-cls-packed-adaln-hard-blind-global-ema-attention-convex.yaml": (
        "centered_normalized_ema_attention"
    ),
}


def load_pretrain_script_module() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wandb_config_tag_bounds_long_config_stems_without_collisions() -> None:
    module = load_pretrain_script_module()
    first_stem = "vit-small-single-cls-packed-adaln-hard-blind-global-ema-attention-convex"
    second_stem = f"{first_stem}-alternate"

    first_tag = module.wandb_config_tag(Path(f"{first_stem}.yaml"))
    second_tag = module.wandb_config_tag(Path(f"{second_stem}.yaml"))

    assert len(first_stem) > module.WANDB_TAG_MAX_LENGTH
    assert len(first_tag) <= module.WANDB_TAG_MAX_LENGTH
    assert len(second_tag) <= module.WANDB_TAG_MAX_LENGTH
    assert first_tag == module.wandb_config_tag(Path(f"{first_stem}.yaml"))
    assert first_tag != second_tag
    assert module.wandb_config_tag(Path("vit-small.yaml")) == "vit-small"


@pytest.mark.parametrize("config_path", PRETRAIN_CONFIG_PATHS)
def test_pretrain_configs_preserve_gram_anchoring_and_predictor_mode(config_path: Path) -> None:
    config = yaml.full_load(config_path.read_text())
    jepa_config = config["jepa"]

    assert isinstance(jepa_config, JEPAConfig)
    assert jepa_config.use_gram_anchoring is True
    assert jepa_config.predictor_attention_mode == "cross_attention"


@pytest.mark.parametrize(("config_path", "expected"), LEJEPA_CONFIGS.items())
def test_lejepa_configs_preserve_shared_target_ablation_boundaries(
    config_path: Path,
    expected: tuple[float | None, str, tuple[int, ...]],
) -> None:
    config = yaml.full_load(config_path.read_text())
    jepa_config = config["jepa"]
    expected_lambda, expected_views, expected_projector_dims = expected

    assert isinstance(jepa_config, JEPAConfig)
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.enable_cls_prediction
    assert jepa_config.lejepa_lambda == expected_lambda
    assert jepa_config.sigreg_views == expected_views
    assert jepa_config.sigreg_projector_dims == expected_projector_dims
    assert jepa_config.sigreg_loss_weight == 0
    assert not jepa_config.use_gram_anchoring
    assert jepa_config.gram_start_epoch is None
    assert jepa_config.cls_prediction_mode == SELECTED_CLS_PREDICTION_MODE


@pytest.mark.parametrize(("config_path", "expected"), LEJEPA_MASKED_OPTIMIZATION_CONFIGS.items())
def test_lejepa_masked_optimization_configs_change_one_mechanism_at_a_time(
    config_path: Path,
    expected: tuple[float, bool, bool],
) -> None:
    config = yaml.full_load(config_path.read_text())
    backbone_config = config["backbone"]
    jepa_config = config["jepa"]
    expected_lambda, expected_cls_prediction, expected_deterministic = expected

    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(jepa_config, JEPAConfig)
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.lejepa_lambda == expected_lambda
    assert jepa_config.sigreg_views == "both"
    assert jepa_config.sigreg_features == "cls_patch_mean"
    assert jepa_config.sigreg_projector_dims == ()
    assert jepa_config.sigreg_loss_weight == 0
    assert jepa_config.enable_cls_prediction is expected_cls_prediction
    assert jepa_config.cls_prediction_mode == (
        SELECTED_CLS_PREDICTION_MODE if expected_cls_prediction else "legacy_cross_attention"
    )
    stochastic_rates = (
        backbone_config.attention_dropout,
        backbone_config.hidden_dropout,
        backbone_config.drop_path_rate,
    )
    assert stochastic_rates == ((0.0, 0.0, 0.0) if expected_deterministic else (0.1, 0.1, 0.1))


@pytest.mark.parametrize(("config_path", "expected"), LEJEPA_MULTIVIEW_CONFIGS.items())
def test_lejepa_multiview_configs_preserve_masked_task_and_preregistered_ladder(
    config_path: Path,
    expected: tuple[int, int, float, float, tuple[int, ...]],
) -> None:
    config = yaml.full_load(config_path.read_text())
    trainer_config = config["trainer"]
    backbone_config = config["backbone"]
    jepa_config = config["jepa"]
    multi_crop = config["multi_crop"]
    expected_global_views, expected_local_views, expected_lambda, expected_weight, expected_projector = expected

    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(jepa_config, JEPAConfig)
    assert trainer_config.batch_size * trainer_config.accumulate_grad_batches == MULTIVIEW_EFFECTIVE_BATCH_SIZE
    assert trainer_config.num_epochs == 100
    assert multi_crop["global_views"] == expected_global_views
    assert multi_crop["local_views"] == expected_local_views
    assert multi_crop["global_scale"] == [0.75, 1.0]
    assert multi_crop["local_scale"] == [0.30, 0.75]
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.enable_cls_prediction
    assert jepa_config.cls_prediction_mode == SELECTED_CLS_PREDICTION_MODE
    assert jepa_config.context_ratio == 0.5
    assert jepa_config.target_ratio == 0.25
    assert jepa_config.scale == 2
    assert jepa_config.sigreg_views == "both"
    assert jepa_config.sigreg_features == "cls_patch_mean"
    assert jepa_config.lejepa_lambda == expected_lambda
    assert jepa_config.invariance_loss_weight == expected_weight
    assert jepa_config.sigreg_projector_dims == expected_projector
    assert not jepa_config.use_gram_anchoring
    assert (backbone_config.attention_dropout, backbone_config.hidden_dropout, backbone_config.drop_path_rate) == (
        0.0,
        0.0,
        0.0,
    )


@pytest.mark.parametrize(("config_path", "expected"), LEJEPA_CONVERGENCE_CONFIGS.items())
def test_lejepa_convergence_configs_isolate_optimizer_and_calibrate_detached_probe(
    config_path: Path,
    expected: tuple[float, float, bool],
) -> None:
    config = yaml.full_load(config_path.read_text())
    backbone_config = config["backbone"]
    jepa_config = config["jepa"]
    optimizer_config = config["optimizer"]
    expected_lr, expected_weight_decay, expected_scheduled = expected

    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(jepa_config, JEPAConfig)
    assert isinstance(optimizer_config, OptimizerConfig)
    assert backbone_config.heads["cls"].dropout == 0
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.lejepa_lambda == 0.10
    assert jepa_config.invariance_loss_weight == 2.0
    assert config["multi_crop"]["global_views"] == 2
    assert config["multi_crop"]["local_views"] == 2
    assert optimizer_config.lr == expected_lr
    assert optimizer_config.weight_decay == expected_weight_decay
    assert optimizer_config.scheduled is expected_scheduled
    assert optimizer_config.parameter_groups[-1] == {
        "params": ["heads"],
        "lr": 0.01,
        "weight_decay": 0.000001,
    }
    if expected_scheduled:
        assert optimizer_config.pct_start == 0.10
        assert optimizer_config.final_div_factor == 100


@pytest.mark.parametrize(("config_path", "expected"), LEJEPA_LOSS_VIEW_CONFIGS.items())
def test_lejepa_loss_view_configs_preserve_optimizer_and_factorial_boundaries(
    config_path: Path,
    expected: tuple[int, int, float, float, bool, list[float]],
) -> None:
    config = yaml.full_load(config_path.read_text())
    backbone_config = config["backbone"]
    jepa_config = config["jepa"]
    optimizer_config = config["optimizer"]
    multi_crop = config["multi_crop"]
    expected_globals, expected_locals, expected_lambda, expected_weight, expected_aux, expected_local_scale = expected

    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(jepa_config, JEPAConfig)
    assert isinstance(optimizer_config, OptimizerConfig)
    assert backbone_config.heads["cls"].dropout == 0
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.enable_cls_prediction is expected_aux
    assert jepa_config.cls_prediction_mode == (
        SELECTED_CLS_PREDICTION_MODE if expected_aux else "legacy_cross_attention"
    )
    assert jepa_config.lejepa_lambda == expected_lambda
    assert jepa_config.invariance_loss_weight == expected_weight
    assert jepa_config.sigreg_views == "both"
    assert jepa_config.sigreg_features == "cls_patch_mean"
    assert jepa_config.sigreg_projector_dims == ()
    assert jepa_config.context_ratio == 0.5
    assert jepa_config.target_ratio == 0.25
    assert jepa_config.scale == 2
    assert multi_crop == {
        "global_views": expected_globals,
        "local_views": expected_locals,
        "global_scale": [0.75, 1.0],
        "local_scale": expected_local_scale,
    }
    assert optimizer_config.lr == 0.002
    assert optimizer_config.weight_decay == 0.2
    assert not optimizer_config.scheduled
    assert optimizer_config.parameter_groups[-1] == {
        "params": ["heads"],
        "lr": 0.01,
        "weight_decay": 0.000001,
    }


@pytest.mark.parametrize(("config_path", "expected"), LEJEPA_TOKEN_DIVERSITY_CONFIGS.items())
def test_lejepa_token_diversity_configs_isolate_preregistered_objectives(
    config_path: Path,
    expected: tuple[str, float, float, int],
) -> None:
    config = yaml.full_load(config_path.read_text())
    trainer_config = config["trainer"]
    backbone_config = config["backbone"]
    jepa_config = config["jepa"]
    optimizer_config = config["optimizer"]
    expected_features, expected_residual_weight, expected_jepa_weight, expected_epochs = expected

    assert isinstance(backbone_config, ViTConfig)
    assert isinstance(jepa_config, JEPAConfig)
    assert isinstance(optimizer_config, OptimizerConfig)
    assert trainer_config.num_epochs == expected_epochs
    assert trainer_config.check_val_every_n_epoch == (5 if expected_epochs == 40 else 1)
    assert trainer_config.batch_size * trainer_config.accumulate_grad_batches == MULTIVIEW_EFFECTIVE_BATCH_SIZE
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.enable_cls_prediction
    assert jepa_config.cls_prediction_mode == SELECTED_CLS_PREDICTION_MODE
    assert jepa_config.lejepa_lambda == 0.05
    assert jepa_config.invariance_loss_weight == 4.0
    assert jepa_config.sigreg_views == "both"
    assert jepa_config.sigreg_features == expected_features
    assert jepa_config.patch_residual_sigreg_loss_weight == expected_residual_weight
    assert jepa_config.patch_residual_sigreg_num_slices == 32
    assert jepa_config.jepa_loss_weight == expected_jepa_weight
    assert config["multi_crop"] == {
        "global_views": 2,
        "local_views": 2,
        "global_scale": [0.75, 1.0],
        "local_scale": [0.30, 0.75],
    }
    assert optimizer_config.lr == 0.002
    assert optimizer_config.weight_decay == 0.2
    assert not optimizer_config.scheduled
    assert optimizer_config.parameter_groups[-1] == {
        "params": ["heads"],
        "lr": 0.01,
        "weight_decay": 0.000001,
    }


@pytest.mark.parametrize(("config_path", "expected"), LEJEPA_PATCH_RANK_CONFIGS.items())
def test_lejepa_patch_rank_configs_isolate_cross_sample_regularization(
    config_path: Path,
    expected: tuple[float, float, int],
) -> None:
    config = yaml.full_load(config_path.read_text())
    trainer_config = config["trainer"]
    jepa_config = config["jepa"]
    expected_lambda, expected_sample_weight, expected_epochs = expected

    assert isinstance(jepa_config, JEPAConfig)
    assert trainer_config.num_epochs == expected_epochs
    assert trainer_config.check_val_every_n_epoch == (5 if expected_epochs == 40 else 1)
    assert trainer_config.batch_size * trainer_config.accumulate_grad_batches == MULTIVIEW_EFFECTIVE_BATCH_SIZE
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.lejepa_lambda == expected_lambda
    assert jepa_config.invariance_loss_weight == 4.0
    assert jepa_config.sigreg_views == "both"
    assert jepa_config.sigreg_features == "cls_patch_mean"
    assert jepa_config.sigreg_projector_dims == ()
    assert jepa_config.patch_residual_sigreg_loss_weight == 0.05
    assert jepa_config.patch_residual_sigreg_num_slices == 32
    assert jepa_config.patch_sample_sigreg_loss_weight == expected_sample_weight
    assert jepa_config.patch_sample_sigreg_num_slices == 32
    assert config["multi_crop"] == {
        "global_views": 2,
        "local_views": 2,
        "global_scale": [0.75, 1.0],
        "local_scale": [0.30, 0.75],
    }


@pytest.mark.parametrize(("config_path", "expected_epochs"), LEJEPA_TARGET_STOPGRAD_CONFIGS.items())
def test_lejepa_target_stopgrad_configs_change_only_gradient_boundary_and_horizon(
    config_path: Path,
    expected_epochs: int,
) -> None:
    baseline = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-token-patch-residual-40e.yaml").read_text()
    )
    candidate = yaml.full_load(config_path.read_text())
    baseline_jepa = baseline["jepa"]
    candidate_jepa = candidate["jepa"]

    assert isinstance(baseline_jepa, JEPAConfig)
    assert isinstance(candidate_jepa, JEPAConfig)
    assert candidate_jepa.target_encoder_mode == "shared"
    assert candidate_jepa.stop_gradient_target
    assert not baseline_jepa.stop_gradient_target
    for field in fields(JEPAConfig):
        if field.name != "stop_gradient_target":
            assert getattr(candidate_jepa, field.name) == getattr(baseline_jepa, field.name)
    assert candidate["trainer"].num_epochs == expected_epochs
    assert candidate["trainer"].check_val_every_n_epoch == (5 if expected_epochs == 40 else 1)
    assert candidate["trainer"].batch_size == baseline["trainer"].batch_size
    assert candidate["trainer"].accumulate_grad_batches == baseline["trainer"].accumulate_grad_batches
    for section in ("backbone", "multi_crop", "optimizer"):
        assert candidate[section] == baseline[section]


@pytest.mark.parametrize(
    ("config_path", "expected_stop_gradient", "expected_epochs"),
    tuple(
        (config_path, expected[0], expected[1]) for config_path, expected in LEJEPA_SINGLE_VIEW_STOPGRAD_CONFIGS.items()
    ),
)
def test_lejepa_single_view_stopgrad_configs_are_crop_free_mim_controls(
    config_path: Path,
    expected_stop_gradient: bool,
    expected_epochs: int,
) -> None:
    config = yaml.full_load(config_path.read_text())
    trainer_config = config["trainer"]
    jepa_config = config["jepa"]

    assert isinstance(jepa_config, JEPAConfig)
    assert trainer_config.num_epochs == expected_epochs
    assert trainer_config.check_val_every_n_epoch == (5 if expected_epochs == 40 else 1)
    assert trainer_config.batch_size * trainer_config.accumulate_grad_batches == MULTIVIEW_EFFECTIVE_BATCH_SIZE
    assert jepa_config.target_encoder_mode == "shared"
    assert jepa_config.stop_gradient_target is expected_stop_gradient
    assert jepa_config.enable_cls_prediction
    assert jepa_config.lejepa_lambda == 0.05
    assert jepa_config.invariance_loss_weight == 0.0
    assert jepa_config.sigreg_views == "both"
    assert jepa_config.sigreg_features == "cls_patch_mean"
    assert jepa_config.patch_residual_sigreg_loss_weight == 0.05
    assert jepa_config.context_ratio == 0.5
    assert jepa_config.target_ratio == 0.25
    assert jepa_config.scale == 2
    assert config["multi_crop"] == {
        "global_views": 1,
        "local_views": 0,
        "global_scale": [0.75, 1.0],
        "local_scale": [0.30, 0.75],
        "use_random_resized_crop": False,
    }


def test_lejepa_single_view_scientific_configs_change_only_target_gradient_boundary() -> None:
    shared_config = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-shared-40e.yaml").read_text()
    )
    stopped_config = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-stopgrad-40e.yaml").read_text()
    )
    shared_jepa = shared_config["jepa"]
    stopped_jepa = stopped_config["jepa"]

    assert isinstance(shared_jepa, JEPAConfig)
    assert isinstance(stopped_jepa, JEPAConfig)
    for field in fields(JEPAConfig):
        if field.name == "stop_gradient_target":
            assert not getattr(shared_jepa, field.name)
            assert getattr(stopped_jepa, field.name)
        else:
            assert getattr(shared_jepa, field.name) == getattr(stopped_jepa, field.name)
    for section in ("trainer", "backbone", "multi_crop", "optimizer"):
        assert shared_config[section] == stopped_config[section]


@pytest.mark.parametrize(
    ("config_path", "changed_field", "expected_value"),
    tuple(
        (config_path, expected[0], expected[1])
        for config_path, expected in LEJEPA_SINGLE_VIEW_BOTTLENECK_CONFIGS.items()
    ),
)
def test_lejepa_single_view_bottleneck_configs_change_exactly_one_objective_weight(
    config_path: Path,
    changed_field: str,
    expected_value: float,
) -> None:
    control = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-lejepa-single-view-stopgrad-40e.yaml").read_text()
    )
    candidate = yaml.full_load(config_path.read_text())
    control_jepa = control["jepa"]
    candidate_jepa = candidate["jepa"]

    assert isinstance(control_jepa, JEPAConfig)
    assert isinstance(candidate_jepa, JEPAConfig)
    assert getattr(candidate_jepa, changed_field) == expected_value
    for field in fields(JEPAConfig):
        if field.name != changed_field:
            assert getattr(candidate_jepa, field.name) == getattr(control_jepa, field.name)
    for section in ("trainer", "backbone", "multi_crop", "optimizer"):
        assert candidate[section] == control[section]


def test_vit_small_defaults_to_selected_partitioned_single_cls_design() -> None:
    pretrain_config = yaml.full_load(DEFAULT_PRETRAIN_CONFIG_PATH.read_text())
    finetune_config = yaml.full_load(DEFAULT_FINETUNE_CONFIG_PATH.read_text())
    selected_pretrain_config = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-partitioned-independent.yaml").read_text()
    )
    selected_finetune_config = yaml.full_load(
        (REPO_ROOT / "config" / "finetune" / "vit-small-single-cls.yaml").read_text()
    )

    assert pretrain_config["backbone"].num_cls_tokens == SELECTED_NUM_CLS_TOKENS
    assert pretrain_config["backbone"].num_register_tokens == SELECTED_NUM_REGISTER_TOKENS
    assert pretrain_config["jepa"].cls_context_tokens == SELECTED_CLS_CONTEXT_TOKENS
    assert pretrain_config["jepa"].cls_prediction_mode == SELECTED_CLS_PREDICTION_MODE
    assert finetune_config["backbone"].num_cls_tokens == SELECTED_NUM_CLS_TOKENS
    assert finetune_config["backbone"].num_register_tokens == SELECTED_NUM_REGISTER_TOKENS
    assert pretrain_config == selected_pretrain_config
    assert finetune_config == selected_finetune_config


@pytest.mark.parametrize(("candidate_path", "control_and_qkv_blocks"), TOKEN_SPECIALIZATION_CONFIGS.items())
def test_token_specialization_configs_change_only_backbone_specialization(
    candidate_path: Path,
    control_and_qkv_blocks: tuple[Path, int],
) -> None:
    control_path, expected_qkv_blocks = control_and_qkv_blocks
    control = yaml.full_load(control_path.read_text())
    candidate = yaml.full_load(candidate_path.read_text())
    control_backbone = control["backbone"]
    candidate_backbone = candidate["backbone"]

    assert isinstance(control_backbone, ViTConfig)
    assert isinstance(candidate_backbone, ViTConfig)
    assert candidate_backbone.specialize_global_token_norms
    assert candidate_backbone.specialize_global_token_qkv_blocks == expected_qkv_blocks
    for field in fields(ViTConfig):
        if field.name not in ("specialize_global_token_norms", "specialize_global_token_qkv_blocks"):
            assert getattr(candidate_backbone, field.name) == getattr(control_backbone, field.name)
    for section in ("trainer", "jepa", "optimizer"):
        assert candidate[section] == control[section]


@pytest.mark.parametrize("study_id", HISTORICAL_FOUR_CLS_STUDY_IDS)
def test_completed_studies_pin_the_historical_four_cls_baseline(study_id: str) -> None:
    spec = yaml.safe_load((REPO_ROOT / "research" / "studies" / f"{study_id}.yaml").read_text())

    assert spec["baseline"]["config"] == "config/pretrain/vit-small-four-cls-legacy.yaml"
    if spec["baseline"].get("finetune_config") is not None:
        assert spec["baseline"]["finetune_config"] == "config/finetune/vit-small-four-cls-legacy.yaml"


@pytest.mark.parametrize(("config_path", "expected_width"), SRELU_WIDTH_CONFIGS.items())
def test_srelu_width_configs_change_only_mlp_fields(config_path: Path, expected_width: int) -> None:
    baseline = yaml.full_load(LEGACY_PRETRAIN_CONFIG_PATH.read_text())
    candidate = yaml.full_load(config_path.read_text())
    baseline_backbone = baseline["backbone"]
    candidate_backbone = candidate["backbone"]

    assert isinstance(baseline_backbone, ViTConfig)
    assert isinstance(candidate_backbone, ViTConfig)
    assert candidate_backbone.activation == "srelu"
    assert candidate_backbone.ffn_hidden_size == expected_width
    assert candidate_backbone.hidden_dropout == pytest.approx(baseline_backbone.hidden_dropout)
    for field in fields(ViTConfig):
        if field.name not in SRELU_CHANGED_BACKBONE_FIELDS:
            assert getattr(candidate_backbone, field.name) == getattr(baseline_backbone, field.name)
    for section in ("trainer", "jepa", "optimizer"):
        assert candidate[section] == baseline[section]


@pytest.mark.parametrize(("config_path", "expected_bias"), SRELU_BIAS_CONFIGS.items())
def test_srelu_bias_configs_change_only_fc1_bias_initialization(config_path: Path, expected_bias: float) -> None:
    baseline = yaml.full_load((REPO_ROOT / "config" / "pretrain" / "vit-small-srelu-h1536.yaml").read_text())
    candidate = yaml.full_load(config_path.read_text())
    baseline_backbone = baseline["backbone"]
    candidate_backbone = candidate["backbone"]

    assert isinstance(baseline_backbone, ViTConfig)
    assert isinstance(candidate_backbone, ViTConfig)
    assert candidate["historical_mlp_fc1_bias_init"] == pytest.approx(expected_bias)
    assert set(candidate) == {*baseline, "historical_mlp_fc1_bias_init"}
    for field in fields(ViTConfig):
        assert getattr(candidate_backbone, field.name) == getattr(baseline_backbone, field.name)
    for section in ("trainer", "jepa", "optimizer"):
        assert candidate[section] == baseline[section]


@pytest.mark.parametrize(("config_path", "cls_prediction_mode"), CLS_CONFIGS.items())
def test_cls_configs_change_only_cls_count_and_prediction_mode(
    config_path: Path,
    cls_prediction_mode: str,
) -> None:
    baseline = yaml.full_load(LEGACY_PRETRAIN_CONFIG_PATH.read_text())
    candidate = yaml.full_load(config_path.read_text())

    assert candidate["backbone"].num_cls_tokens == 1
    assert candidate["jepa"].cls_prediction_mode == cls_prediction_mode
    for field in fields(ViTConfig):
        if field.name != "num_cls_tokens":
            assert getattr(candidate["backbone"], field.name) == getattr(baseline["backbone"], field.name)
    for field in fields(JEPAConfig):
        if field.name != "cls_prediction_mode":
            assert getattr(candidate["jepa"], field.name) == getattr(baseline["jepa"], field.name)
    for section in ("trainer", "optimizer"):
        assert candidate[section] == baseline[section]


@pytest.mark.parametrize(("config_path", "cls_prediction_mode"), CLS_REGISTER_CONFIGS.items())
def test_cls_register_configs_reclassify_three_cls_tokens_and_change_only_prediction_mode(
    config_path: Path,
    cls_prediction_mode: str,
) -> None:
    baseline = yaml.full_load(LEGACY_PRETRAIN_CONFIG_PATH.read_text())
    candidate = yaml.full_load(config_path.read_text())

    assert candidate["backbone"].num_cls_tokens == 1
    assert candidate["backbone"].num_register_tokens == 7
    assert candidate["backbone"].num_cls_tokens + candidate["backbone"].num_register_tokens == (
        baseline["backbone"].num_cls_tokens + baseline["backbone"].num_register_tokens
    )
    assert candidate["jepa"].cls_prediction_mode == cls_prediction_mode
    for field in fields(ViTConfig):
        if field.name not in ("num_cls_tokens", "num_register_tokens"):
            assert getattr(candidate["backbone"], field.name) == getattr(baseline["backbone"], field.name)
    for field in fields(JEPAConfig):
        if field.name != "cls_prediction_mode":
            assert getattr(candidate["jepa"], field.name) == getattr(baseline["jepa"], field.name)
    for section in ("trainer", "optimizer"):
        assert candidate[section] == baseline[section]


@pytest.mark.parametrize(("config_path", "cls_context_tokens"), CLS_PARTITION_COUNT_CONFIGS.items())
def test_cls_partition_count_configs_change_only_partition_count(
    config_path: Path,
    cls_context_tokens: int,
) -> None:
    control = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-partitioned-independent.yaml").read_text()
    )
    candidate = yaml.full_load(config_path.read_text())

    assert candidate["jepa"].cls_context_tokens == cls_context_tokens
    for field in fields(JEPAConfig):
        if field.name != "cls_context_tokens":
            assert getattr(candidate["jepa"], field.name) == getattr(control["jepa"], field.name)
    for section in ("trainer", "backbone", "optimizer"):
        assert candidate[section] == control[section]


@pytest.mark.parametrize(
    ("config_path", "expected"),
    CLS_JOINT_CONTEXT_CONFIGS.items(),
)
def test_cls_joint_context_configs_change_only_prediction_mode_and_single_pass_loss_weight(
    config_path: Path,
    expected: tuple[str, float],
) -> None:
    control = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-partitioned-independent.yaml").read_text()
    )
    candidate = yaml.full_load(config_path.read_text())

    assert candidate["backbone"].num_cls_tokens == 1
    assert candidate["backbone"].num_register_tokens == 7
    cls_prediction_mode, jepa_loss_weight = expected
    assert candidate["jepa"].cls_prediction_mode == cls_prediction_mode
    assert candidate["jepa"].jepa_loss_weight == pytest.approx(jepa_loss_weight)
    for field in fields(JEPAConfig):
        if field.name not in ("cls_prediction_mode", "jepa_loss_weight"):
            assert getattr(candidate["jepa"], field.name) == getattr(control["jepa"], field.name)
    for section in ("trainer", "backbone", "optimizer"):
        assert candidate[section] == control[section]


def test_cls_joint_context_smoke_uses_maximum_routing_granularity() -> None:
    config = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "smoke-cls-joint-context-token-routed.yaml").read_text()
    )

    assert config["trainer"].num_epochs == 1
    assert config["backbone"].num_cls_tokens == 1
    assert config["backbone"].num_register_tokens == 7
    assert config["jepa"].cls_prediction_mode == "joint_context_token_routed"
    assert config["jepa"].jepa_loss_weight == pytest.approx(2.0)


def test_cls_packed_dual_routing_smoke_exercises_query_duplication() -> None:
    config = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "smoke-cls-joint-context-packed-dual-routed.yaml").read_text()
    )

    assert config["trainer"].num_epochs == 1
    assert config["backbone"].num_cls_tokens == 1
    assert config["backbone"].num_register_tokens == 7
    assert config["jepa"].cls_prediction_mode == "joint_context_packed_dual_routed"
    assert config["jepa"].jepa_loss_weight == pytest.approx(1.0)


@pytest.mark.parametrize(("config_path", "cls_prediction_mode"), CLS_PACKED_ADALN_CONFIGS.items())
def test_cls_packed_adaln_configs_change_only_prediction_mode(
    config_path: Path,
    cls_prediction_mode: str,
) -> None:
    control = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-register-partitioned-independent.yaml").read_text()
    )
    candidate = yaml.full_load(config_path.read_text())

    assert candidate["jepa"].cls_prediction_mode == cls_prediction_mode
    for field in fields(JEPAConfig):
        if field.name != "cls_prediction_mode":
            assert getattr(candidate["jepa"], field.name) == getattr(control["jepa"], field.name)
    for section in ("trainer", "backbone", "optimizer"):
        assert candidate[section] == control[section]


def test_cls_packed_adaln_smoke_exercises_hard_blind_packing() -> None:
    config = yaml.full_load((REPO_ROOT / "config" / "pretrain" / "smoke-cls-packed-adaln-hard-blind.yaml").read_text())

    assert config["trainer"].num_epochs == 1
    assert config["backbone"].num_cls_tokens == 1
    assert config["backbone"].num_register_tokens == 7
    assert config["jepa"].cls_prediction_mode == "packed_adaln_hard_blind"
    assert config["jepa"].jepa_loss_weight == pytest.approx(1.0)


@pytest.mark.parametrize(("config_path", "expected_weight"), CLS_GLOBAL_TARGET_CONFIGS.items())
def test_cls_global_target_configs_change_only_global_loss_weight(
    config_path: Path,
    expected_weight: float,
) -> None:
    blinded = yaml.full_load((REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind.yaml").read_text())
    candidate = yaml.full_load(config_path.read_text())

    assert candidate["cls_global_target_loss_weight"] == pytest.approx(expected_weight)
    assert set(candidate) == {*blinded, "cls_global_target_loss_weight"}
    for section in ("trainer", "backbone", "jepa", "optimizer"):
        assert candidate[section] == blinded[section]


def test_cls_global_target_configuration_requires_blinded_single_cls_model() -> None:
    pretrain_script = load_pretrain_script_module()
    baseline = yaml.full_load(LEGACY_PRETRAIN_CONFIG_PATH.read_text())
    blinded = yaml.full_load((REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind.yaml").read_text())

    pretrain_script.validate_cls_global_target_configuration(blinded["backbone"], blinded["jepa"], 0.1)
    pretrain_script.validate_cls_global_target_configuration(baseline["backbone"], baseline["jepa"], 0.0)
    with pytest.raises(ValueError, match="exactly one student CLS token"):
        pretrain_script.validate_cls_global_target_configuration(baseline["backbone"], blinded["jepa"], 0.1)
    with pytest.raises(ValueError, match="visually blinded"):
        pretrain_script.validate_cls_global_target_configuration(blinded["backbone"], baseline["jepa"], 0.1)


@pytest.mark.parametrize("loss_weight", (-0.1, float("inf"), float("nan")))
def test_cls_global_target_configuration_rejects_invalid_weight(loss_weight: float) -> None:
    pretrain_script = load_pretrain_script_module()
    blinded = yaml.full_load((REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind.yaml").read_text())

    with pytest.raises(ValueError, match="finite non-negative"):
        pretrain_script.validate_cls_global_target_configuration(blinded["backbone"], blinded["jepa"], loss_weight)


@pytest.mark.parametrize(("config_path", "expected_pooling"), CLS_TEACHER_GLOBAL_CONFIGS.items())
def test_cls_teacher_global_configs_change_only_pooled_target_objective(
    config_path: Path,
    expected_pooling: str,
) -> None:
    control = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind.yaml").read_text()
    )
    candidate = yaml.full_load(config_path.read_text())

    assert candidate["cls_global_target_pooling"] == expected_pooling
    assert candidate["cls_global_target_loss_weight"] == pytest.approx(0.1)
    assert candidate["cls_global_pool_consistency_loss_weight"] == pytest.approx(0.1)
    assert set(candidate) == {
        *control,
        "cls_global_target_pooling",
        "cls_global_target_loss_weight",
        "cls_global_pool_consistency_loss_weight",
    }
    for section in ("trainer", "backbone", "jepa", "optimizer"):
        assert candidate[section] == control[section]


def test_cls_teacher_global_configuration_accepts_hard_blind_normalized_pooling() -> None:
    pretrain_script = load_pretrain_script_module()
    hard_blind = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind.yaml").read_text()
    )

    pretrain_script.validate_cls_global_target_configuration(
        hard_blind["backbone"],
        hard_blind["jepa"],
        0.1,
        "centered_normalized_mean",
        0.1,
    )
    pretrain_script.validate_cls_global_target_configuration(
        hard_blind["backbone"],
        hard_blind["jepa"],
        0.1,
        "centered_normalized_ema_attention",
        0.1,
    )


def test_cls_teacher_global_configuration_requires_online_loss_for_ema_pooler() -> None:
    pretrain_script = load_pretrain_script_module()
    hard_blind = yaml.full_load(
        (REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-packed-adaln-hard-blind.yaml").read_text()
    )

    with pytest.raises(ValueError, match="positive pool-consistency"):
        pretrain_script.validate_cls_global_target_configuration(
            hard_blind["backbone"],
            hard_blind["jepa"],
            0.1,
            "centered_normalized_ema_attention",
            0.0,
        )
    with pytest.raises(ValueError, match="positive pool-consistency"):
        pretrain_script.validate_cls_global_target_configuration(
            hard_blind["backbone"],
            hard_blind["jepa"],
            0.0,
            "centered_normalized_ema_attention",
            0.0,
        )


def test_historical_single_cls_finetune_config_changes_only_cls_count() -> None:
    baseline = yaml.full_load(LEGACY_FINETUNE_CONFIG_PATH.read_text())
    single_cls = yaml.full_load(LEGACY_SINGLE_CLS_FINETUNE_CONFIG_PATH.read_text())

    assert single_cls["backbone"].num_cls_tokens == SELECTED_NUM_CLS_TOKENS
    for field in fields(ViTConfig):
        if field.name != "num_cls_tokens":
            assert getattr(single_cls["backbone"], field.name) == getattr(baseline["backbone"], field.name)
    for section in ("trainer", "optimizer"):
        assert single_cls[section] == baseline[section]


def test_instantiate_jepa_forwards_predictor_configuration(mocker) -> None:
    pretrain_script = load_pretrain_script_module()
    backbone = object()
    backbone_config = SimpleNamespace(instantiate=mocker.Mock(return_value=backbone))
    jepa_config = SimpleNamespace(
        predictor_depth=3,
        predictor_attention_mode="decoder",
        cls_prediction_mode="legacy_cross_attention",
        cls_context_tokens=4,
        disable_predictor_regularizers=True,
    )
    device = object()
    predictor = object()
    model = object()
    predictor_constructor = mocker.patch.object(pretrain_script, "CrossAttentionPredictor", return_value=predictor)
    model_constructor = mocker.patch.object(pretrain_script, "CIFAR10MJEPA", return_value=model)

    result = pretrain_script.instantiate_jepa(backbone_config, jepa_config, device)

    assert result is model
    backbone_config.instantiate.assert_called_once_with(device=device)
    predictor_constructor.assert_called_once_with(
        backbone,
        jepa_config.predictor_depth,
        device=device,
        attention_mode=jepa_config.predictor_attention_mode,
        cls_prediction_mode=jepa_config.cls_prediction_mode,
        cls_context_tokens=jepa_config.cls_context_tokens,
        disable_predictor_regularizers=jepa_config.disable_predictor_regularizers,
    )
    model_constructor.assert_called_once_with(
        jepa_config,
        backbone,
        predictor,
        cls_global_target_pooling="raw_mean",
    )


def test_pretrain_checkpoint_argument_is_forwarded_to_full_state_restore(mocker, tmp_path: Path) -> None:
    pretrain_script = load_pretrain_script_module()
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = CheckpointMetadata(
        step=17,
        epoch=4,
        img_size=(32, 32),
        elapsed_seconds=91.25,
        wandb_run_id="wandb-123",
    )
    jepa = SimpleNamespace(student=object(), predictor=object(), teacher=object())
    optimizer = object()
    scheduler = object()
    load_checkpoint = mocker.patch.object(pretrain_script, "load_checkpoint", return_value=(17, 4))

    resume = pretrain_script.restore_pretraining_checkpoint(
        checkpoint,
        metadata,
        jepa,
        optimizer,
        scheduler,
        requested_wandb_run_id=None,
    )

    load_checkpoint.assert_called_once_with(
        checkpoint,
        jepa.student,
        jepa.predictor,
        jepa.teacher,
        optimizer,
        scheduler,
    )
    assert resume == (17, 4, 91.25, "wandb-123")


def test_pretrain_resume_preserves_launch_cls_path_benchmark(tmp_path: Path) -> None:
    pretrain_script = load_pretrain_script_module()

    assert pretrain_script.should_benchmark_cls_prediction_path(None) is True
    assert pretrain_script.should_benchmark_cls_prediction_path(tmp_path / "checkpoint.pt") is False


def test_pretrain_resume_rejects_different_wandb_run_id(mocker, tmp_path: Path) -> None:
    pretrain_script = load_pretrain_script_module()
    metadata = CheckpointMetadata(17, 4, (32, 32), 91.25, "original")
    load_checkpoint = mocker.patch.object(pretrain_script, "load_checkpoint")

    with pytest.raises(ValueError, match="does not match checkpoint"):
        pretrain_script.restore_pretraining_checkpoint(
            tmp_path / "checkpoint.pt",
            metadata,
            SimpleNamespace(student=object(), predictor=object(), teacher=object()),
            object(),
            object(),
            requested_wandb_run_id="different",
        )

    load_checkpoint.assert_not_called()


def test_pretrain_parser_accepts_seed_and_checkpoint(mocker) -> None:
    pretrain_script = load_pretrain_script_module()
    mocker.patch.object(
        sys,
        "argv",
        ["pretrain.py", "config.yaml", "/data", "--checkpoint", "checkpoint.pt", "--seed", "2"],
    )

    args = pretrain_script.parse_args()

    assert args.checkpoint == Path("checkpoint.pt")
    assert args.seed == 2


def test_managed_lifecycle_reporter_requires_matching_supervisor_environment(tmp_path: Path) -> None:
    pretrain_script = load_pretrain_script_module()
    arguments = SimpleNamespace(study_id="study-a", name="run-a")
    environment = {
        "MJEPA_RESEARCH_STUDY_ID": "study-a",
        "MJEPA_RESEARCH_RUN_ID": "run-a",
        "MJEPA_RESEARCH_ATTEMPT": "2",
        "MJEPA_RESEARCH_THREAD_ID": "thread-a",
    }

    reporter = pretrain_script.build_managed_lifecycle_reporter(arguments, tmp_path, environment)

    assert reporter is not None
    assert reporter.study_id == "study-a"
    assert reporter.run_id == "run-a"
    assert reporter.attempt == 2


def test_managed_lifecycle_reporter_is_disabled_for_unmanaged_training(tmp_path: Path) -> None:
    pretrain_script = load_pretrain_script_module()

    assert (
        pretrain_script.build_managed_lifecycle_reporter(
            SimpleNamespace(study_id=None, name="manual"),
            tmp_path,
            {},
        )
        is None
    )


def test_resume_applies_checkpoint_image_size_before_model_construction() -> None:
    pretrain_script = load_pretrain_script_module()
    backbone_config = pretrain_script.ViTConfig(
        in_channels=3,
        hidden_size=64,
        patch_size=[4, 4],
        img_size=[256, 256],
        depth=2,
        num_attention_heads=4,
        ffn_hidden_size=128,
    )
    metadata = CheckpointMetadata(17, 4, (32, 32), 91.25, "wandb-123")

    resumed_config = pretrain_script.apply_checkpoint_image_size(backbone_config, metadata)

    assert list(resumed_config.img_size) == [32, 32]
