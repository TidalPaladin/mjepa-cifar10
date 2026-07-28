import math
import os
import socket
from collections.abc import Callable
from contextlib import closing
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics as tm
from mjepa import CLSPredictionMode, JEPAConfig
from mjepa.jepa import (
    ADALN_BLIND_CLS_PREDICTION_MODE,
    JOINT_CONTEXT_CLS_PREDICTION_MODE,
    PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODE,
    PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    PROJECTED_CLS_PREDICTION_MODE,
    SLOT_BIAS_CLS_PREDICTION_MODE,
    SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    CrossAttentionPredictor,
    compute_jepa_prediction_loss,
)
from mjepa.metrics import CLSPatchAlignmentMetric
from mjepa.model import MJEPAPredictions
from mjepa.optimizer import OptimizerConfig
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from vit import AttentivePoolHeadConfig, HeadConfig, ViTConfig, ViTFeatures

import mjepa_cifar10.pretrain as pretrain_module
from mjepa_cifar10.pretrain import (
    CIFAR10MJEPA,
    CPA_RESULT_KEYS,
    OptimizerStepResult,
    clip_optimizer_grad_norm_,
    compute_and_reset_cpa_metrics,
    compute_and_reset_mean_percentage,
    compute_cls_aux_shuffle_diagnostic,
    compute_cls_global_target_diagnostic,
    compute_cls_global_target_loss,
    did_gradient_clip,
    get_gradient_norm_stats,
    get_gradient_sync_context,
    get_scheduler_last_lr,
    report_checkpoint_lifecycle,
    run_optimizer_step,
    update_cls_patch_alignment_metric,
)


def test_get_scheduler_last_lr_returns_first_learning_rate() -> None:
    scheduler = RecordingScheduler([], learning_rates=[0.2, 0.1])

    assert get_scheduler_last_lr(scheduler) == 0.2


def test_report_checkpoint_lifecycle_emits_first_cycle_once() -> None:
    progress_events: list[tuple[str, int, int, float]] = []
    first_cycle_events: list[tuple[int, int, float]] = []

    first_cycle_reported = report_checkpoint_lifecycle(
        progress_callback=lambda phase, epoch, step, seconds: progress_events.append((phase, epoch, step, seconds)),
        first_cycle_callback=lambda epoch, step, seconds: first_cycle_events.append((epoch, step, seconds)),
        validation_completed=True,
        first_cycle_reported=False,
        epoch=0,
        optimizer_step=10,
        active_seconds=12.5,
    )
    first_cycle_reported = report_checkpoint_lifecycle(
        progress_callback=lambda phase, epoch, step, seconds: progress_events.append((phase, epoch, step, seconds)),
        first_cycle_callback=lambda epoch, step, seconds: first_cycle_events.append((epoch, step, seconds)),
        validation_completed=True,
        first_cycle_reported=first_cycle_reported,
        epoch=1,
        optimizer_step=20,
        active_seconds=25.0,
    )

    assert first_cycle_reported
    assert progress_events == [("checkpointed", 0, 10, 12.5), ("checkpointed", 1, 20, 25.0)]
    assert first_cycle_events == [(0, 10, 12.5)]


FIRST_PARAMETER_GRAD = torch.tensor([3.0, 4.0])
SECOND_PARAMETER_GRAD = torch.tensor([0.0, 12.0])
FIRST_PARAMETER_GRAD_NORM = 5.0
SECOND_PARAMETER_GRAD_NORM = 12.0
EXPECTED_GRAD_NORM_MEAN = (FIRST_PARAMETER_GRAD_NORM + SECOND_PARAMETER_GRAD_NORM) / 2
INITIAL_TRAIN_STEP = 3
TOTAL_TRAIN_STEPS = 10
DDP_DRIFT_TEST_STEPS = 12
DDP_DRIFT_TOLERANCE = 1e-5
DRIFT_TEST_BATCH_SIZE = 4
DDP_DRIFT_CLIP_NORM = 0.05
TINY_BACKBONE_IMG_SIZE = [16, 16]
TINY_BACKBONE_HIDDEN_SIZE = 64
TINY_BACKBONE_FFN_HIDDEN_SIZE = 256
TINY_BACKBONE_DEPTH = 2
TINY_BACKBONE_ATTENTION_HEADS = 4
TINY_BACKBONE_REGISTER_TOKENS = 2
TINY_BACKBONE_CLS_TOKENS = 2
TINY_PREDICTOR_DEPTH = 1


def test_get_gradient_norm_stats_returns_mean_and_max() -> None:
    first_parameter = nn.Parameter(torch.zeros_like(FIRST_PARAMETER_GRAD))
    second_parameter = nn.Parameter(torch.zeros_like(SECOND_PARAMETER_GRAD))
    first_parameter.grad = FIRST_PARAMETER_GRAD.clone()
    second_parameter.grad = SECOND_PARAMETER_GRAD.clone()

    assert get_gradient_norm_stats([first_parameter, second_parameter]) == pytest.approx(
        (EXPECTED_GRAD_NORM_MEAN, SECOND_PARAMETER_GRAD_NORM)
    )


def test_get_gradient_norm_stats_ignores_parameters_without_gradients() -> None:
    parameter_with_grad = nn.Parameter(torch.zeros_like(FIRST_PARAMETER_GRAD))
    parameter_without_grad = nn.Parameter(torch.zeros_like(SECOND_PARAMETER_GRAD))
    parameter_with_grad.grad = FIRST_PARAMETER_GRAD.clone()

    assert get_gradient_norm_stats([parameter_with_grad, parameter_without_grad]) == pytest.approx(
        (FIRST_PARAMETER_GRAD_NORM, FIRST_PARAMETER_GRAD_NORM)
    )


def test_get_gradient_norm_stats_returns_none_without_gradients() -> None:
    parameters = [
        nn.Parameter(torch.zeros_like(FIRST_PARAMETER_GRAD)),
        nn.Parameter(torch.zeros_like(SECOND_PARAMETER_GRAD)),
    ]

    assert get_gradient_norm_stats(parameters) is None


def test_get_gradient_sync_context_uses_no_sync_for_intermediate_microbatches() -> None:
    events: list[str] = []

    class RecordingNoSync:
        def __enter__(self) -> None:
            events.append("no_sync.enter")
            return None

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append("no_sync.exit")
            return None

    with get_gradient_sync_context(lambda: RecordingNoSync(), should_sync_gradients=False):
        events.append("body")

    assert events == ["no_sync.enter", "body", "no_sync.exit"]


def test_get_gradient_sync_context_skips_no_sync_for_final_microbatch() -> None:
    events: list[str] = []

    with get_gradient_sync_context(lambda: pytest.fail("no_sync should not be called"), should_sync_gradients=True):
        events.append("body")

    assert events == ["body"]


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _total_grad_norm_for_optimizer(optimizer: object) -> float:
    param_groups = getattr(optimizer, "param_groups")
    seen_parameter_ids: set[int] = set()
    grad_norms: list[Tensor] = []
    for group in param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids or parameter.grad is None:
                continue
            seen_parameter_ids.add(parameter_id)
            grad_norms.append(parameter.grad.detach().norm(2))
    if not grad_norms:
        return 0.0
    return torch.stack(grad_norms).norm(2).item()


def _ddp_hybrid_muon_clip_sync_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        torch.cuda.set_device(rank)
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.LayerNorm(4),
            nn.GELU(),
            nn.Linear(4, 2),
        ).cuda(rank)
        ddp_model = DDP(model, device_ids=[rank])
        optimizer_config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind="hybrid_muon",
            scheduled=False,
            fused=False,
            max_grad_norm=FIRST_PARAMETER_GRAD_NORM,
        )
        optimizer, scheduler = optimizer_config.instantiate(ddp_model, total_steps=DDP_DRIFT_TEST_STEPS)

        microbatches = (
            (
                torch.tensor([[1.0 + rank, 2.0 + rank, 3.0 + rank, 4.0 + rank]], dtype=torch.float32, device=rank),
                torch.tensor([[0.5, -0.5]], dtype=torch.float32, device=rank),
            ),
            (
                torch.tensor([[5.0 + rank, 6.0 + rank, 7.0 + rank, 8.0 + rank]], dtype=torch.float32, device=rank),
                torch.tensor([[1.5, -1.5]], dtype=torch.float32, device=rank),
            ),
        )

        for index, (features, target) in enumerate(microbatches):
            should_sync = index == len(microbatches) - 1
            with get_gradient_sync_context(ddp_model.no_sync, should_sync):
                prediction = ddp_model(features)
                loss = torch.nn.functional.mse_loss(prediction, target)
                loss.backward()

        optimizer_step_result = run_optimizer_step(
            optimizer,
            scheduler,
            step=0,
            total_steps=1,
            max_grad_norm=FIRST_PARAMETER_GRAD_NORM,
        )
        assert optimizer_step_result.next_step == 1

        for parameter in ddp_model.module.parameters():
            averaged_parameter = parameter.detach().clone()
            dist.all_reduce(averaged_parameter, op=dist.ReduceOp.AVG)
            assert torch.allclose(parameter.detach(), averaged_parameter, atol=1e-6, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.ci_skip
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires 2 CUDA devices")
@pytest.mark.skipif(not dist.is_nccl_available(), reason="NCCL is unavailable")
@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
def test_ddp_hybrid_muon_gradient_clipping_keeps_parameters_synced() -> None:
    world_size = 2
    port = _find_free_port()

    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _ddp_hybrid_muon_clip_sync_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )


def _ddp_cifar10_mjepa_clip_sync_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        torch.random.manual_seed(0)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        backbone_config = ViTConfig(
            in_channels=3,
            hidden_size=TINY_BACKBONE_HIDDEN_SIZE,
            patch_size=[4, 4],
            img_size=TINY_BACKBONE_IMG_SIZE,
            depth=TINY_BACKBONE_DEPTH,
            num_attention_heads=TINY_BACKBONE_ATTENTION_HEADS,
            ffn_hidden_size=TINY_BACKBONE_FFN_HIDDEN_SIZE,
            num_register_tokens=TINY_BACKBONE_REGISTER_TOKENS,
            num_cls_tokens=TINY_BACKBONE_CLS_TOKENS,
            dtype=torch.float32,
            heads={
                "cls": HeadConfig(
                    out_features=pretrain_module.NUM_CLASSES,
                )
            },
        )
        jepa_config = JEPAConfig(gram_start_epoch=None, predictor_depth=TINY_PREDICTOR_DEPTH, scale=2)
        optimizer_config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind="hybrid_muon",
            scheduled=False,
            fused=False,
            max_grad_norm=DDP_DRIFT_CLIP_NORM,
        )
        backbone = backbone_config.instantiate(device=device)
        predictor = CrossAttentionPredictor(backbone, depth=jepa_config.predictor_depth, device=device)
        model = CIFAR10MJEPA(jepa_config, backbone, predictor)
        ddp_model = DDP(model, device_ids=[rank])
        optimizer, scheduler = optimizer_config.instantiate(ddp_model, total_steps=DDP_DRIFT_TEST_STEPS)
        epoch = 0
        channels = backbone_config.in_channels
        image_height, image_width = backbone_config.img_size
        base_images = torch.arange(
            DRIFT_TEST_BATCH_SIZE * channels * image_height * image_width,
            dtype=torch.float32,
            device=device,
        ).view(DRIFT_TEST_BATCH_SIZE, channels, image_height, image_width)
        max_student_drift = 0.0
        max_predictor_drift = 0.0

        for step in range(DDP_DRIFT_TEST_STEPS):
            for microbatch_index in range(2):
                image_offset = step * 20 + microbatch_index * 10 + rank
                label_offset = (step * DRIFT_TEST_BATCH_SIZE + microbatch_index * 2) % 10
                images = base_images + image_offset
                labels = torch.tensor(
                    [(label_offset + index) % 10 for index in range(DRIFT_TEST_BATCH_SIZE)],
                    dtype=torch.long,
                    device=device,
                )
                should_sync = microbatch_index == 1
                with get_gradient_sync_context(ddp_model.no_sync, should_sync):
                    output = ddp_model(images, jepa_config.scale, epoch)
                    ssl_losses = ddp_model.module.compute_losses(output, step, epoch)
                    probe_loss = torch.nn.functional.cross_entropy(output.probes["cls"], labels)
                    loss = ssl_losses.reduce() + probe_loss
                    loss.backward()

            grad_norm_before = _total_grad_norm_for_optimizer(optimizer)
            assert grad_norm_before > DDP_DRIFT_CLIP_NORM
            optimizer_step_result = run_optimizer_step(
                optimizer,
                scheduler,
                step=step,
                total_steps=DDP_DRIFT_TEST_STEPS,
                max_grad_norm=optimizer_config.max_grad_norm,
                update_teacher=lambda current_step=step: ddp_model.module.update_teacher(
                    current_step, DDP_DRIFT_TEST_STEPS
                ),
            )
            assert optimizer_step_result.next_step == step + 1
            assert optimizer_step_result.grad_clip_triggered

            for parameter in ddp_model.module.student.parameters():
                averaged_parameter = parameter.detach().clone()
                dist.all_reduce(averaged_parameter, op=dist.ReduceOp.AVG)
                drift = torch.max(torch.abs(parameter.detach() - averaged_parameter)).item()
                max_student_drift = max(max_student_drift, drift)

            for parameter in ddp_model.module.predictor.parameters():
                averaged_parameter = parameter.detach().clone()
                dist.all_reduce(averaged_parameter, op=dist.ReduceOp.AVG)
                drift = torch.max(torch.abs(parameter.detach() - averaged_parameter)).item()
                max_predictor_drift = max(max_predictor_drift, drift)

        assert max_student_drift <= DDP_DRIFT_TOLERANCE, (
            f"student drift exceeded tolerance: max_student_drift={max_student_drift:.8f}, "
            f"tolerance={DDP_DRIFT_TOLERANCE:.8f}"
        )
        assert max_predictor_drift <= DDP_DRIFT_TOLERANCE, (
            f"predictor drift exceeded tolerance: max_predictor_drift={max_predictor_drift:.8f}, "
            f"tolerance={DDP_DRIFT_TOLERANCE:.8f}"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.ci_skip
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires 2 CUDA devices")
@pytest.mark.skipif(not dist.is_nccl_available(), reason="NCCL is unavailable")
@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
def test_ddp_cifar10_mjepa_gradient_clipping_does_not_accumulate_rank_drift() -> None:
    world_size = 2
    port = _find_free_port()

    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _ddp_cifar10_mjepa_clip_sync_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )


def test_clip_optimizer_grad_norm_ignores_duplicate_parameters() -> None:
    parameter = nn.Parameter(torch.zeros_like(SECOND_PARAMETER_GRAD))
    parameter.grad = SECOND_PARAMETER_GRAD.clone()
    optimizer = RecordingOptimizer([])
    optimizer.param_groups = [{"params": [parameter, parameter]}]

    clipped_norm = clip_optimizer_grad_norm_(optimizer, max_grad_norm=FIRST_PARAMETER_GRAD_NORM)

    assert clipped_norm is not None
    assert clipped_norm.item() == pytest.approx(SECOND_PARAMETER_GRAD_NORM)
    assert parameter.grad.norm().item() == pytest.approx(FIRST_PARAMETER_GRAD_NORM)


def test_clip_optimizer_grad_norm_returns_none_when_disabled() -> None:
    optimizer = RecordingOptimizer([])
    optimizer.param_groups = [{"params": []}]

    assert clip_optimizer_grad_norm_(optimizer, max_grad_norm=None) is None


@pytest.mark.parametrize(
    ("total_grad_norm", "max_grad_norm", "expected"),
    [
        (torch.tensor(FIRST_PARAMETER_GRAD_NORM - 0.1), FIRST_PARAMETER_GRAD_NORM, False),
        (torch.tensor(FIRST_PARAMETER_GRAD_NORM), FIRST_PARAMETER_GRAD_NORM, False),
        (torch.tensor(FIRST_PARAMETER_GRAD_NORM + 0.1), FIRST_PARAMETER_GRAD_NORM, True),
        (torch.tensor(FIRST_PARAMETER_GRAD_NORM + 0.1), None, False),
        (None, FIRST_PARAMETER_GRAD_NORM, False),
    ],
)
def test_did_gradient_clip_matches_threshold(
    total_grad_norm: Tensor | None,
    max_grad_norm: float | None,
    expected: bool,
) -> None:
    assert did_gradient_clip(total_grad_norm, max_grad_norm) is expected


def test_compute_and_reset_mean_percentage_uses_independent_logging_blocks() -> None:
    metric = tm.MeanMetric()

    metric.update(1.0)
    metric.update(0.0)
    metric.update(1.0)
    assert compute_and_reset_mean_percentage(metric) == pytest.approx(66.66666666666666)

    metric.update(0.0)
    metric.update(0.0)
    metric.update(1.0)
    assert compute_and_reset_mean_percentage(metric) == pytest.approx(33.33333333333333)


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
    count_state = cast(Tensor, metric.count)
    hist_state = cast(Tensor, metric.hist)
    torch.testing.assert_close(count_state, torch.zeros_like(count_state))
    torch.testing.assert_close(hist_state.sum(), torch.zeros_like(hist_state.sum()))


def test_compute_and_reset_cpa_metrics_prefixes_keys_and_resets_state() -> None:
    metric = CLSPatchAlignmentMetric(num_bins=4096)
    cls_tokens = torch.tensor([[1.0, 0.0]])
    patch_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    metric.update(cls_tokens, patch_tokens)
    expected_metrics = {key: value.item() for key, value in metric.compute().items()}

    logged_metrics = compute_and_reset_cpa_metrics(metric, prefix="train")

    assert logged_metrics == {f"train/{key}": pytest.approx(value) for key, value in expected_metrics.items()}
    assert tuple(key.removeprefix("train/") for key in logged_metrics) == CPA_RESULT_KEYS
    count_state = cast(Tensor, metric.count)
    hist_state = cast(Tensor, metric.hist)
    torch.testing.assert_close(count_state, torch.zeros_like(count_state))
    torch.testing.assert_close(hist_state.sum(), torch.zeros_like(hist_state.sum()))


class RecordingScheduler:
    def __init__(self, events: list[str], learning_rates: list[float] | None = None):
        self.events = events
        self.learning_rates = learning_rates or []

    def step(self, epoch: int | None = None) -> None:
        del epoch
        self.events.append("scheduler.step")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    def get_last_lr(self) -> list[float]:
        return self.learning_rates


class RecordingOptimizer:
    def __init__(self, events: list[str]):
        self.events = events
        self.param_groups: list[dict[str, Any]] = []

    def step(self, closure: Callable[[], float] | None = None) -> None:
        del closure
        self.events.append("optimizer.step")

    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none
        self.events.append("optimizer.zero_grad")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


def test_run_optimizer_step_calls_clip_hook_before_optimizer_step(mocker) -> None:
    events: list[str] = []
    scheduler = RecordingScheduler(events)
    optimizer = RecordingOptimizer(events)
    clip_grad_norm = mocker.patch.object(pretrain_module, "clip_optimizer_grad_norm_")

    def clip_side_effect(clipped_optimizer, max_grad_norm: float | None) -> Tensor:
        assert clipped_optimizer is optimizer
        assert max_grad_norm == FIRST_PARAMETER_GRAD_NORM
        events.append("clip_grad_norm")
        return torch.tensor(FIRST_PARAMETER_GRAD_NORM + 0.1)

    clip_grad_norm.side_effect = clip_side_effect

    def update_teacher() -> None:
        events.append("update_teacher")

    optimizer_step_result = run_optimizer_step(
        optimizer,
        scheduler,
        INITIAL_TRAIN_STEP,
        TOTAL_TRAIN_STEPS,
        max_grad_norm=FIRST_PARAMETER_GRAD_NORM,
        update_teacher=update_teacher,
    )

    assert optimizer_step_result == OptimizerStepResult(
        next_step=INITIAL_TRAIN_STEP + 1,
        grad_clip_triggered=True,
    )
    assert events == [
        "clip_grad_norm",
        "scheduler.step",
        "optimizer.step",
        "optimizer.zero_grad",
        "update_teacher",
    ]


def test_run_optimizer_step_skips_scheduler_after_total_steps() -> None:
    events: list[str] = []
    scheduler = RecordingScheduler(events)
    optimizer = RecordingOptimizer(events)

    optimizer_step_result = run_optimizer_step(
        optimizer,
        scheduler,
        TOTAL_TRAIN_STEPS,
        TOTAL_TRAIN_STEPS,
    )

    assert optimizer_step_result == OptimizerStepResult(
        next_step=TOTAL_TRAIN_STEPS + 1,
        grad_clip_triggered=False,
    )
    assert events == ["optimizer.step", "optimizer.zero_grad"]


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
    head_config: HeadConfig | AttentivePoolHeadConfig | None = None,
    cls_prediction_mode: CLSPredictionMode = "legacy_cross_attention",
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
    predictor = CrossAttentionPredictor(backbone, depth=1, cls_prediction_mode=cls_prediction_mode)
    return CIFAR10MJEPA(
        JEPAConfig(gram_start_epoch=None, cls_prediction_mode=cls_prediction_mode),
        backbone,
        predictor,
        autocast_dtype=torch.float32,
    )


def make_features(*, num_cls_tokens: int) -> ViTFeatures:
    cls_count = num_cls_tokens
    total_tokens = cls_count + NUM_REGISTER_TOKENS + NUM_VISUAL_TOKENS
    dense_features = torch.arange(BATCH_SIZE * total_tokens * HIDDEN_SIZE, dtype=torch.float32).view(
        BATCH_SIZE, total_tokens, HIDDEN_SIZE
    )
    return ViTFeatures(dense_features, NUM_REGISTER_TOKENS, cls_count, tokenized_size=(2, 2))


@pytest.mark.parametrize(
    ("cls_prediction_mode", "num_cls_tokens"),
    (
        ("legacy_cross_attention", NUM_CLS_TOKENS),
        (ADALN_BLIND_CLS_PREDICTION_MODE, 1),
        (PROJECTED_CLS_PREDICTION_MODE, 1),
        (SLOT_BIAS_CLS_PREDICTION_MODE, 1),
    ),
)
def test_cls_aux_shuffle_diagnostic_blinds_cross_sample_identity(
    mocker,
    cls_prediction_mode: CLSPredictionMode,
    num_cls_tokens: int,
) -> None:
    model = make_model(num_cls_tokens=num_cls_tokens, cls_prediction_mode=cls_prediction_mode)
    student_output = make_features(num_cls_tokens=num_cls_tokens)
    teacher_output = make_features(num_cls_tokens=num_cls_tokens)
    target_mask = torch.ones(BATCH_SIZE, NUM_VISUAL_TOKENS, dtype=torch.bool)
    true_prediction = torch.zeros(BATCH_SIZE, NUM_VISUAL_TOKENS, HIDDEN_SIZE)
    shuffled_prediction = torch.ones_like(true_prediction)
    predictions = MJEPAPredictions(
        pred=true_prediction,
        pred_with_cls=true_prediction,
        student_output=student_output,
        teacher_output=teacher_output,
        context_mask=torch.zeros_like(target_mask),
        target_mask=target_mask,
    )
    recompute = mocker.patch.object(
        model,
        "forward_cls_predictor",
        side_effect=(true_prediction, shuffled_prediction),
    )

    metrics = compute_cls_aux_shuffle_diagnostic(model, predictions)

    target = teacher_output.visual_tokens
    expected_true = compute_jepa_prediction_loss(true_prediction, target).item()
    expected_shuffled = compute_jepa_prediction_loss(shuffled_prediction, target).item()
    assert metrics == pytest.approx(
        {
            "pretrain/validation_cls_aux_loss": expected_true,
            "pretrain/validation_cls_aux_loss_shuffled": expected_shuffled,
            "pretrain/validation_cls_aux_shuffle_gap": expected_shuffled - expected_true,
        }
    )
    shuffled_cls = torch.roll(student_output.cls_tokens, shifts=1, dims=0)
    assert recompute.call_count == 2
    assert torch.equal(recompute.call_args_list[1].args[1], shuffled_cls)


def test_joint_context_shuffle_diagnostic_isolates_cls_and_visual_sources(mocker) -> None:
    model = make_model(num_cls_tokens=1, cls_prediction_mode=JOINT_CONTEXT_CLS_PREDICTION_MODE)
    student_output = make_features(num_cls_tokens=1)
    teacher_output = make_features(num_cls_tokens=1)
    context_mask = torch.ones(BATCH_SIZE, NUM_VISUAL_TOKENS, dtype=torch.bool)
    target_mask = torch.ones_like(context_mask)
    joint_prediction = torch.zeros(BATCH_SIZE, NUM_VISUAL_TOKENS, HIDDEN_SIZE)
    shuffled_joint_prediction = torch.ones_like(joint_prediction)
    cls_only_prediction = torch.full_like(joint_prediction, 2.0)
    shuffled_cls_only_prediction = torch.full_like(joint_prediction, 3.0)
    visual_only_prediction = torch.full_like(joint_prediction, 4.0)
    predictions = MJEPAPredictions(
        pred=joint_prediction,
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=teacher_output,
        context_mask=context_mask,
        target_mask=target_mask,
    )
    recompute = mocker.patch.object(
        model,
        "forward_predictor",
        side_effect=(
            joint_prediction,
            shuffled_joint_prediction,
            cls_only_prediction,
            shuffled_cls_only_prediction,
            visual_only_prediction,
        ),
    )

    metrics = compute_cls_aux_shuffle_diagnostic(model, predictions)

    target = teacher_output.visual_tokens
    joint_loss = compute_jepa_prediction_loss(joint_prediction, target).item()
    shuffled_joint_loss = compute_jepa_prediction_loss(shuffled_joint_prediction, target).item()
    cls_only_loss = compute_jepa_prediction_loss(cls_only_prediction, target).item()
    shuffled_cls_only_loss = compute_jepa_prediction_loss(shuffled_cls_only_prediction, target).item()
    visual_only_loss = compute_jepa_prediction_loss(visual_only_prediction, target).item()
    assert metrics == pytest.approx(
        {
            "pretrain/validation_cls_aux_loss": cls_only_loss,
            "pretrain/validation_cls_aux_loss_shuffled": shuffled_cls_only_loss,
            "pretrain/validation_cls_aux_shuffle_gap": shuffled_cls_only_loss - cls_only_loss,
            "pretrain/validation_joint_context_loss": joint_loss,
            "pretrain/validation_joint_context_loss_shuffled_cls": shuffled_joint_loss,
            "pretrain/validation_joint_context_cls_shuffle_gap": shuffled_joint_loss - joint_loss,
            "pretrain/validation_visual_only_loss": visual_only_loss,
        }
    )
    assert recompute.call_count == 5
    shuffled_cls = torch.roll(student_output.cls_tokens, shifts=1, dims=0)
    assert torch.equal(recompute.call_args_list[0].args[1][:, -1:], student_output.cls_tokens)
    assert torch.equal(recompute.call_args_list[1].args[1][:, -1:], shuffled_cls)
    joint_mask = recompute.call_args_list[0].kwargs["context_attention_mask"]
    cls_only_mask = recompute.call_args_list[2].kwargs["context_attention_mask"]
    visual_only_mask = recompute.call_args_list[4].kwargs["context_attention_mask"]
    assert joint_mask.all()
    assert not cls_only_mask[..., :-1].any()
    assert cls_only_mask[..., -1].all()
    assert visual_only_mask[..., :-1].all()
    assert not visual_only_mask[..., -1].any()


def test_packed_joint_context_shuffle_diagnostic_aligns_duplicated_queries(mocker) -> None:
    model = make_model(
        num_cls_tokens=1,
        cls_prediction_mode=PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    )
    student_output = make_features(num_cls_tokens=1)
    teacher_output = make_features(num_cls_tokens=1)
    context_mask = torch.ones(BATCH_SIZE, NUM_VISUAL_TOKENS, dtype=torch.bool)
    target_mask = torch.ones_like(context_mask)
    prediction = torch.zeros(BATCH_SIZE, 2 * NUM_VISUAL_TOKENS, HIDDEN_SIZE)
    predictions = MJEPAPredictions(
        pred=prediction,
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=teacher_output,
        context_mask=context_mask,
        target_mask=target_mask,
    )
    mocker.patch.object(model, "forward_predictor", return_value=prediction)

    metrics = compute_cls_aux_shuffle_diagnostic(model, predictions)

    repeated_target = teacher_output.visual_tokens.repeat(1, 2, 1)
    expected_loss = compute_jepa_prediction_loss(prediction, repeated_target).item()
    assert metrics["pretrain/validation_joint_context_loss"] == pytest.approx(expected_loss)


def test_packed_adaln_shuffle_diagnostic_isolates_blind_cls_dependence(mocker) -> None:
    model = make_model(
        num_cls_tokens=1,
        cls_prediction_mode=PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODE,
    )
    student_output = make_features(num_cls_tokens=1)
    teacher_output = make_features(num_cls_tokens=1)
    context_mask = torch.ones(BATCH_SIZE, NUM_VISUAL_TOKENS, dtype=torch.bool)
    target_mask = torch.ones_like(context_mask)
    visual_prediction = torch.zeros(BATCH_SIZE, NUM_VISUAL_TOKENS, HIDDEN_SIZE)
    blind_prediction = torch.full_like(visual_prediction, 2.0)
    shuffled_blind_prediction = torch.full_like(visual_prediction, 3.0)
    true_prediction = torch.cat([visual_prediction, blind_prediction], dim=1)
    shuffled_prediction = torch.cat([visual_prediction, shuffled_blind_prediction], dim=1)
    predictions = MJEPAPredictions(
        pred=true_prediction,
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=teacher_output,
        context_mask=context_mask,
        target_mask=target_mask,
    )
    recompute = mocker.patch.object(
        model,
        "forward_packed_adaln_hard_blind_predictor_heads",
        side_effect=((true_prediction, None), (shuffled_prediction, None)),
    )

    metrics = compute_cls_aux_shuffle_diagnostic(model, predictions)

    target = teacher_output.visual_tokens
    visual_loss = compute_jepa_prediction_loss(visual_prediction, target).item()
    blind_loss = compute_jepa_prediction_loss(blind_prediction, target).item()
    shuffled_blind_loss = compute_jepa_prediction_loss(shuffled_blind_prediction, target).item()
    assert metrics == pytest.approx(
        {
            "pretrain/validation_cls_aux_loss": blind_loss,
            "pretrain/validation_cls_aux_loss_shuffled": shuffled_blind_loss,
            "pretrain/validation_cls_aux_shuffle_gap": shuffled_blind_loss - blind_loss,
            "pretrain/validation_visual_only_loss": visual_loss,
        }
    )
    assert recompute.call_count == 2
    shuffled_cls = torch.roll(student_output.cls_tokens, shifts=1, dims=0)
    assert torch.equal(recompute.call_args_list[0].args[2], student_output.cls_tokens)
    assert torch.equal(recompute.call_args_list[1].args[2], shuffled_cls)


def test_source_balanced_joint_diagnostic_applies_cls_cardinality_bias() -> None:
    model = make_model(
        num_cls_tokens=1,
        cls_prediction_mode=SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    )
    student_output = make_features(num_cls_tokens=1)
    predictions = MJEPAPredictions(
        pred=torch.zeros(BATCH_SIZE, NUM_VISUAL_TOKENS, HIDDEN_SIZE),
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=make_features(num_cls_tokens=1),
        context_mask=torch.ones(BATCH_SIZE, NUM_VISUAL_TOKENS, dtype=torch.bool),
        target_mask=torch.ones(BATCH_SIZE, NUM_VISUAL_TOKENS, dtype=torch.bool),
    )

    source_mask = pretrain_module._joint_context_source_mask(
        predictions,
        cls_prediction_mode=model.config.cls_prediction_mode,
        show_visual=True,
        show_cls=True,
    )

    assert source_mask.dtype == predictions.pred.dtype
    assert (source_mask[..., :-1] == 0).all()
    assert torch.equal(
        source_mask[..., -1],
        torch.full_like(source_mask[..., -1], math.log(NUM_VISUAL_TOKENS)),
    )


def test_cls_global_target_loss_regresses_one_student_cls_to_pooled_teacher_visual_tokens() -> None:
    student_output = make_features(num_cls_tokens=1)
    teacher_output = make_features(num_cls_tokens=1)
    predictions = MJEPAPredictions(
        pred=torch.empty(0),
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=teacher_output,
        context_mask=torch.empty(0, dtype=torch.bool),
        target_mask=torch.empty(0, dtype=torch.bool),
    )

    loss = compute_cls_global_target_loss(predictions)

    expected_target = teacher_output.visual_tokens.mean(dim=1)
    assert loss == pytest.approx(torch.nn.functional.mse_loss(student_output.cls_tokens[:, 0], expected_target))


def test_cls_global_target_loss_does_not_read_student_visual_or_register_tokens() -> None:
    student_dense = torch.randn(
        BATCH_SIZE,
        1 + NUM_REGISTER_TOKENS + NUM_VISUAL_TOKENS,
        HIDDEN_SIZE,
        requires_grad=True,
    )
    student_output = ViTFeatures(
        student_dense,
        NUM_REGISTER_TOKENS,
        num_cls_tokens=1,
        tokenized_size=(2, 2),
    )
    teacher_output = make_features(num_cls_tokens=1)
    predictions = MJEPAPredictions(
        pred=torch.empty(0),
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=teacher_output,
        context_mask=torch.empty(0, dtype=torch.bool),
        target_mask=torch.empty(0, dtype=torch.bool),
    )

    compute_cls_global_target_loss(predictions).backward()

    assert student_dense.grad is not None
    assert torch.count_nonzero(student_dense.grad[:, 0]).item() > 0
    assert torch.count_nonzero(student_dense.grad[:, 1:]).item() == 0


def test_cls_global_target_diagnostic_compares_true_and_cross_sample_shuffled_cls() -> None:
    student_output = make_features(num_cls_tokens=1)
    teacher_output = make_features(num_cls_tokens=1)
    predictions = MJEPAPredictions(
        pred=torch.empty(0),
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=teacher_output,
        context_mask=torch.empty(0, dtype=torch.bool),
        target_mask=torch.empty(0, dtype=torch.bool),
    )

    metrics = compute_cls_global_target_diagnostic(predictions)

    teacher_target = teacher_output.visual_tokens.mean(dim=1)
    student_cls = student_output.cls_tokens[:, 0]
    expected_true = torch.nn.functional.mse_loss(student_cls, teacher_target).item()
    expected_shuffled = torch.nn.functional.mse_loss(torch.roll(student_cls, shifts=1, dims=0), teacher_target).item()
    assert metrics == pytest.approx(
        {
            "pretrain/validation_cls_global_loss": expected_true,
            "pretrain/validation_cls_global_loss_shuffled": expected_shuffled,
            "pretrain/validation_cls_global_shuffle_gap": expected_shuffled - expected_true,
            "pretrain/validation_cls_global_student_norm": student_cls.norm(dim=-1).mean().item(),
            "pretrain/validation_cls_global_teacher_norm": teacher_target.norm(dim=-1).mean().item(),
        }
    )


def test_cls_global_target_loss_requires_exactly_one_student_cls_token() -> None:
    student_output = make_features(num_cls_tokens=NUM_CLS_TOKENS)
    predictions = MJEPAPredictions(
        pred=torch.empty(0),
        pred_with_cls=None,
        student_output=student_output,
        teacher_output=make_features(num_cls_tokens=NUM_CLS_TOKENS),
        context_mask=torch.empty(0, dtype=torch.bool),
        target_mask=torch.empty(0, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="exactly one student CLS token"):
        compute_cls_global_target_loss(predictions)


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


def test_probe_loss_updates_only_classifier_head() -> None:
    model = make_model(
        num_cls_tokens=NUM_CLS_TOKENS,
        head_config=HeadConfig(out_features=10),
    )
    images = torch.randn(BATCH_SIZE, 3, 8, 8)
    labels = torch.tensor([0, 1])

    teacher_features = model.forward_teacher(images)
    logits = model.forward_probe(teacher_features)["cls"]
    torch.nn.functional.cross_entropy(logits, labels).backward()

    head_parameter_ids = {id(parameter) for parameter in model.student.get_head("cls").parameters()}
    assert head_parameter_ids
    for parameter in model.student.parameters():
        if id(parameter) in head_parameter_ids:
            assert parameter.grad is not None
        else:
            assert parameter.grad is None
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
