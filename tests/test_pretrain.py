import os
import socket
from contextlib import closing
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics as tm
from mjepa import JEPAConfig
from mjepa.jepa import CrossAttentionPredictor
from mjepa.metrics import CLSPatchAlignmentMetric
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
    did_gradient_clip,
    get_gradient_norm_stats,
    get_gradient_sync_context,
    get_scheduler_last_lr,
    run_optimizer_step,
    update_cls_patch_alignment_metric,
)


def test_get_scheduler_last_lr_returns_first_learning_rate() -> None:
    scheduler = SimpleNamespace(get_last_lr=lambda: [0.2, 0.1])

    assert get_scheduler_last_lr(scheduler) == 0.2


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

    mp.spawn(
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

    mp.spawn(
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
    assert metric.count.item() == 0
    assert metric.hist.sum().item() == 0.0


def test_compute_and_reset_cpa_metrics_prefixes_keys_and_resets_state() -> None:
    metric = CLSPatchAlignmentMetric(num_bins=4096)
    cls_tokens = torch.tensor([[1.0, 0.0]])
    patch_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    metric.update(cls_tokens, patch_tokens)
    expected_metrics = {key: value.item() for key, value in metric.compute().items()}

    logged_metrics = compute_and_reset_cpa_metrics(metric, prefix="train")

    assert logged_metrics == {f"train/{key}": pytest.approx(value) for key, value in expected_metrics.items()}
    assert tuple(key.removeprefix("train/") for key in logged_metrics) == CPA_RESULT_KEYS
    assert metric.count.item() == 0
    assert metric.hist.sum().item() == 0.0


class RecordingScheduler:
    def __init__(self, events: list[str]):
        self.events = events

    def step(self) -> None:
        self.events.append("scheduler.step")


class RecordingOptimizer:
    def __init__(self, events: list[str]):
        self.events = events

    def step(self) -> None:
        self.events.append("optimizer.step")

    def zero_grad(self) -> None:
        self.events.append("optimizer.zero_grad")


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
