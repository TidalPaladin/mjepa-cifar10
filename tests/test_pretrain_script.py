import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from mjepa import JEPAConfig
from mjepa.trainer import CheckpointMetadata
from vit import ViTConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_CONFIG_PATHS = (
    REPO_ROOT / "config" / "pretrain" / "vit-small.yaml",
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
}
CLS_GLOBAL_TARGET_CONFIGS = {
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind-global-w0p1.yaml": 0.1,
    REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind-global-w0p5.yaml": 0.5,
}


def load_pretrain_script_module() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain_script_module", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("config_path", PRETRAIN_CONFIG_PATHS)
def test_pretrain_configs_preserve_gram_anchoring_and_predictor_mode(config_path: Path) -> None:
    config = yaml.full_load(config_path.read_text())
    jepa_config = config["jepa"]

    assert isinstance(jepa_config, JEPAConfig)
    assert jepa_config.use_gram_anchoring is True
    assert jepa_config.predictor_attention_mode == "cross_attention"


@pytest.mark.parametrize(("config_path", "expected_width"), SRELU_WIDTH_CONFIGS.items())
def test_srelu_width_configs_change_only_mlp_fields(config_path: Path, expected_width: int) -> None:
    baseline = yaml.full_load(PRETRAIN_CONFIG_PATHS[0].read_text())
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
    baseline = yaml.full_load(PRETRAIN_CONFIG_PATHS[0].read_text())
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
    baseline = yaml.full_load(PRETRAIN_CONFIG_PATHS[0].read_text())
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
    baseline = yaml.full_load(PRETRAIN_CONFIG_PATHS[0].read_text())
    blinded = yaml.full_load((REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind.yaml").read_text())

    pretrain_script.validate_cls_global_target_configuration(blinded["backbone"], blinded["jepa"], 0.1)
    pretrain_script.validate_cls_global_target_configuration(baseline["backbone"], baseline["jepa"], 0.0)
    with pytest.raises(ValueError, match="exactly one student CLS token"):
        pretrain_script.validate_cls_global_target_configuration(baseline["backbone"], blinded["jepa"], 0.1)
    with pytest.raises(ValueError, match="adaln_blind"):
        pretrain_script.validate_cls_global_target_configuration(blinded["backbone"], baseline["jepa"], 0.1)


@pytest.mark.parametrize("loss_weight", (-0.1, float("inf"), float("nan")))
def test_cls_global_target_configuration_rejects_invalid_weight(loss_weight: float) -> None:
    pretrain_script = load_pretrain_script_module()
    blinded = yaml.full_load((REPO_ROOT / "config" / "pretrain" / "vit-small-single-cls-adaln-blind.yaml").read_text())

    with pytest.raises(ValueError, match="finite non-negative"):
        pretrain_script.validate_cls_global_target_configuration(blinded["backbone"], blinded["jepa"], loss_weight)


def test_single_cls_finetune_config_changes_only_cls_count() -> None:
    baseline = yaml.full_load((REPO_ROOT / "config" / "finetune" / "vit-small.yaml").read_text())
    single_cls = yaml.full_load((REPO_ROOT / "config" / "finetune" / "vit-small-single-cls.yaml").read_text())

    assert single_cls["backbone"].num_cls_tokens == 1
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
        disable_predictor_regularizers=jepa_config.disable_predictor_regularizers,
    )
    model_constructor.assert_called_once_with(jepa_config, backbone, predictor)


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
