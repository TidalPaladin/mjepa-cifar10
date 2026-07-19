import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from mjepa import JEPAConfig
from mjepa.trainer import CheckpointMetadata


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_CONFIG_PATHS = (
    REPO_ROOT / "config" / "pretrain" / "vit-small.yaml",
    REPO_ROOT / "config" / "pretrain" / "vit-tiny.yaml",
)


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


def test_instantiate_jepa_forwards_predictor_configuration(mocker) -> None:
    pretrain_script = load_pretrain_script_module()
    backbone = object()
    backbone_config = SimpleNamespace(instantiate=mocker.Mock(return_value=backbone))
    jepa_config = SimpleNamespace(
        predictor_depth=3,
        predictor_attention_mode="decoder",
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
