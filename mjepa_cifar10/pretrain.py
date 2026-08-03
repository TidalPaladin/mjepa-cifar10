import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Final, Literal, cast

import torch
import torch.nn.functional as F
import torchmetrics as tm
from mjepa.jepa import (
    ADALN_BLIND_CLS_PREDICTION_MODE,
    JOINT_CONTEXT_CLS_PREDICTION_MODES,
    PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES,
    SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    CLSPredictionMode,
    CrossAttentionPredictor,
    JEPAConfig,
    compute_jepa_prediction_loss,
    get_momentum,
    joint_context_query_multiplier,
    setup_teacher,
)
from mjepa.jepa import (
    update_teacher as update_ema_teacher,
)
from mjepa.metrics import CLSPatchAlignmentMetric
from mjepa.model import MJEPA, MJEPAPredictions
from mjepa.optimizer import OptimizerLike, SchedulerLike
from mjepa.trainer import (
    DataLoaderFn,
    TrainerConfig,
    calculate_total_steps,
    format_pbar_description,
    is_rank_zero,
    rank_zero_info,
    save_checkpoint,
    scale_change,
    should_step_optimizer,
    size_change,
)
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torchmetrics.wrappers import Running
from tqdm import tqdm
from vit import ViT, ViTFeatures

import wandb

from .classification import forward_classifier
from .collapse import (
    EmbeddingCollapseMetric,
    PatchTokenDiversityMetric,
    compute_and_reset_collapse_metrics,
    compute_and_reset_patch_token_diversity_metrics,
)
from .experiment import append_metric_record, save_safetensors_atomic
from .train_utils import (
    OptimizerStepResult,
    clip_optimizer_grad_norm_,
    compute_and_reset_mean_percentage,
    did_gradient_clip,
    get_gradient_norm_stats,
    get_gradient_sync_context,
    get_scheduler_last_lr,
)


NUM_CLASSES: Final[int] = 10
WINDOW: Final[int] = 5
LOG_INTERVAL: Final[int] = 50
VALIDATION_DIAGNOSTIC_SEED: Final[int] = 0
LOSS_DENOMINATOR_EPSILON: Final[float] = 1e-12
DEFAULT_CLS_GLOBAL_TARGET_LOSS_WEIGHT: Final[float] = 0.0
DEFAULT_CLS_GLOBAL_POOL_CONSISTENCY_LOSS_WEIGHT: Final[float] = 0.0
RAW_MEAN_CLS_GLOBAL_TARGET_POOLING: Final[str] = "raw_mean"
CENTERED_NORMALIZED_MEAN_CLS_GLOBAL_TARGET_POOLING: Final[str] = "centered_normalized_mean"
CENTERED_NORMALIZED_EMA_ATTENTION_CLS_GLOBAL_TARGET_POOLING: Final[str] = "centered_normalized_ema_attention"
CLS_GLOBAL_TARGET_POOLINGS: Final[frozenset[str]] = frozenset(
    (
        RAW_MEAN_CLS_GLOBAL_TARGET_POOLING,
        CENTERED_NORMALIZED_MEAN_CLS_GLOBAL_TARGET_POOLING,
        CENTERED_NORMALIZED_EMA_ATTENTION_CLS_GLOBAL_TARGET_POOLING,
    )
)
CLS_GLOBAL_TARGET_POOLER_MODULE_NAME: Final[str] = "_cls_global_target_poolers"
GRAD_CLIP_TRIGGER_PCT_KEY: Final[str] = "pretrain/grad_clip_trigger_pct"
CPA_RESULT_KEYS: Final[tuple[str, ...]] = ("cpa_mean", "cpa_std", "cpa_p90", "cpa_p99")
ProgressPhase = Literal["training", "validation", "checkpointing", "checkpointed"]
ProgressCallback = Callable[[ProgressPhase, int, int, float], object]
FirstCycleCallback = Callable[[int, int, float], object]
__all__ = [
    "AttentionWeightPool",
    "CPA_RESULT_KEYS",
    "CENTERED_NORMALIZED_EMA_ATTENTION_CLS_GLOBAL_TARGET_POOLING",
    "CENTERED_NORMALIZED_MEAN_CLS_GLOBAL_TARGET_POOLING",
    "CIFAR10MJEPA",
    "CLS_GLOBAL_TARGET_POOLINGS",
    "CLSGlobalTargetLosses",
    "DEFAULT_CLS_GLOBAL_TARGET_LOSS_WEIGHT",
    "DEFAULT_CLS_GLOBAL_POOL_CONSISTENCY_LOSS_WEIGHT",
    "GRAD_CLIP_TRIGGER_PCT_KEY",
    "NUM_CLASSES",
    "OptimizerStepResult",
    "RAW_MEAN_CLS_GLOBAL_TARGET_POOLING",
    "clip_optimizer_grad_norm_",
    "compute_and_reset_cpa_metrics",
    "compute_cls_aux_shuffle_diagnostic",
    "compute_and_reset_mean_percentage",
    "did_gradient_clip",
    "compute_cls_global_target_diagnostic",
    "compute_cls_global_target_loss",
    "compute_cls_global_target_objective",
    "compute_visual_target_shuffle_diagnostic",
    "get_gradient_norm_stats",
    "get_gradient_sync_context",
    "get_scheduler_last_lr",
    "run_optimizer_step",
    "report_checkpoint_lifecycle",
    "split_training_views",
    "train",
    "update_cls_patch_alignment_metric",
]


def split_training_views(images: Tensor) -> tuple[Tensor, Tensor | None]:
    """Separate the masked-task anchor from optional independently augmented views."""
    if images.ndim == 4:
        return images, None
    if images.ndim == 5 and images.shape[1] > 1:
        return images[:, 0], images[:, 1:]
    raise ValueError("training images must have shape (B,C,H,W) or (B,V,C,H,W) with V > 1")


def report_checkpoint_lifecycle(
    *,
    progress_callback: ProgressCallback | None,
    first_cycle_callback: FirstCycleCallback | None,
    validation_completed: bool,
    first_cycle_reported: bool,
    epoch: int,
    optimizer_step: int,
    active_seconds: float,
) -> bool:
    """Report a durable checkpoint and the first complete train-validation cycle."""
    if progress_callback is not None:
        progress_callback("checkpointed", epoch, optimizer_step, active_seconds)
    if validation_completed and not first_cycle_reported and first_cycle_callback is not None:
        first_cycle_callback(epoch, optimizer_step, active_seconds)
        return True
    return first_cycle_reported


class AttentionWeightPool(nn.Module):
    """Learn attention weights while keeping pooled values in the input feature space."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        *,
        bias: bool = False,
        qk_normalization: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.query = nn.Parameter(torch.empty(1, num_attention_heads, 1, self.head_dim, **factory_kwargs))
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.query_norm = nn.LayerNorm(self.head_dim, **factory_kwargs) if qk_normalization else nn.Identity()
        self.key_norm = nn.LayerNorm(self.head_dim, **factory_kwargs) if qk_normalization else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.xavier_uniform_(self.key_proj.weight)
        if self.key_proj.bias is not None:
            nn.init.zeros_(self.key_proj.bias)
        for norm in (self.query_norm, self.key_norm):
            if isinstance(norm, nn.LayerNorm):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)

    def forward_weights(self, visual_tokens: Tensor) -> Tensor:
        batch_size, num_tokens, hidden_size = visual_tokens.shape
        if hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"Expected visual-token width {self.num_attention_heads * self.head_dim}, got {hidden_size}"
            )
        keys = self.key_proj(visual_tokens).view(
            batch_size,
            num_tokens,
            self.num_attention_heads,
            self.head_dim,
        )
        keys = self.key_norm(keys.movedim(1, 2))
        query = self.query_norm(self.query)
        logits = (query * keys).sum(dim=-1) * (self.head_dim**-0.5)
        return logits.softmax(dim=-1).mean(dim=1)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        weights = self.forward_weights(visual_tokens)
        return torch.einsum("bt,btd->bd", weights, visual_tokens)


class CLSGlobalTargetPoolers(nn.Module):
    """Matched online and EMA attention poolers for teacher-global targets."""

    def __init__(self, backbone: ViT):
        super().__init__()
        config = backbone.config
        reference_parameter = next(backbone.parameters())
        self.online = AttentionWeightPool(
            config.hidden_size,
            config.num_attention_heads,
            bias=config.attention_bias,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
            qk_normalization=config.qk_normalization,
        )
        self.target = setup_teacher(self.online)


@dataclass(frozen=True)
class CLSGlobalTargetLosses:
    cls_loss: Tensor
    pool_consistency_loss: Tensor | None
    teacher_target: Tensor


class CIFAR10MJEPA(MJEPA):
    def __init__(
        self,
        config: JEPAConfig,
        backbone: ViT,
        predictor: CrossAttentionPredictor,
        autocast_dtype: torch.dtype = torch.bfloat16,
        cls_global_target_pooling: str = RAW_MEAN_CLS_GLOBAL_TARGET_POOLING,
    ):
        super().__init__(config, backbone, predictor, autocast_dtype=autocast_dtype)
        if cls_global_target_pooling not in CLS_GLOBAL_TARGET_POOLINGS:
            raise ValueError(f"Unsupported CLS global-target pooling mode: {cls_global_target_pooling!r}")
        self.cls_global_target_pooling = cls_global_target_pooling
        if cls_global_target_pooling == CENTERED_NORMALIZED_EMA_ATTENTION_CLS_GLOBAL_TARGET_POOLING:
            predictor.add_module(CLS_GLOBAL_TARGET_POOLER_MODULE_NAME, CLSGlobalTargetPoolers(backbone))

    @property
    def cls_global_target_poolers(self) -> CLSGlobalTargetPoolers | None:
        module = getattr(self.predictor, CLS_GLOBAL_TARGET_POOLER_MODULE_NAME, None)
        return cast(CLSGlobalTargetPoolers | None, module)

    def forward_probe(self, features: ViTFeatures) -> dict[str, Tensor]:
        return {"cls": forward_classifier(self.student, features, detach_features=True)}

    def update_teacher(self, step: int, total_steps: int) -> None:
        super().update_teacher(step, total_steps)
        if (poolers := self.cls_global_target_poolers) is not None:
            momentum = get_momentum(step, total_steps, self.config.momentum, self.config.scheduled)
            update_ema_teacher(poolers.online, poolers.target, momentum)


def compute_and_reset_cpa_metrics(metric: CLSPatchAlignmentMetric, prefix: str) -> dict[str, float]:
    cpa_metrics = metric.compute()
    metric.reset()
    return {f"{prefix}/{key}": cpa_metrics[key].item() for key in CPA_RESULT_KEYS}


def update_cls_patch_alignment_metric(metric: CLSPatchAlignmentMetric | None, features: ViTFeatures) -> bool:
    if metric is None or not MJEPA._has_cls_tokens(features):
        return False

    metric.update(features.cls_tokens, features.visual_tokens)
    return True


def _joint_context_source_mask(
    output: MJEPAPredictions,
    *,
    cls_prediction_mode: CLSPredictionMode,
    show_visual: bool,
    show_cls: bool,
) -> Tensor:
    batch_size = output.pred.shape[0]
    target_tokens = int(output.target_mask.sum(dim=1)[0].item()) * joint_context_query_multiplier(cls_prediction_mode)
    visual_context_tokens = output.student_output.visual_tokens.shape[1]
    if (
        cls_prediction_mode == SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE
        and show_visual
        and show_cls
    ):
        mask = torch.zeros(
            batch_size,
            1,
            target_tokens,
            visual_context_tokens + 1,
            dtype=output.pred.dtype,
            device=output.pred.device,
        )
        mask[..., -1] = math.log(visual_context_tokens)
        return mask
    mask = torch.zeros(
        batch_size,
        1,
        target_tokens,
        visual_context_tokens + 1,
        dtype=torch.bool,
        device=output.pred.device,
    )
    mask[..., :-1] = show_visual
    mask[..., -1] = show_cls
    return mask


def _forward_joint_context_diagnostic(
    jepa: MJEPA,
    output: MJEPAPredictions,
    cls_tokens: Tensor,
    source_mask: Tensor,
    tokenized_size: tuple[int, int],
) -> Tensor:
    joint_context = torch.cat([output.student_output.visual_tokens, cls_tokens], dim=1)
    return jepa.forward_predictor(
        tokenized_size,
        joint_context,
        output.context_mask,
        output.target_mask,
        rope_seed=VALIDATION_DIAGNOSTIC_SEED,
        context_attention_mask=source_mask,
    )


def _compute_joint_context_shuffle_diagnostic(
    jepa: MJEPA,
    output: MJEPAPredictions,
    tokenized_size: tuple[int, int],
) -> dict[str, float]:
    cls_tokens = output.student_output.cls_tokens
    if cls_tokens.shape[1] != 1:
        raise ValueError("joint-context CLS diagnostics require exactly one CLS token")
    shuffled_cls_tokens = torch.roll(cls_tokens, shifts=1, dims=0)
    cls_prediction_mode = jepa.config.cls_prediction_mode
    joint_mask = _joint_context_source_mask(
        output,
        cls_prediction_mode=cls_prediction_mode,
        show_visual=True,
        show_cls=True,
    )
    cls_only_mask = _joint_context_source_mask(
        output,
        cls_prediction_mode=cls_prediction_mode,
        show_visual=False,
        show_cls=True,
    )
    visual_only_mask = _joint_context_source_mask(
        output,
        cls_prediction_mode=cls_prediction_mode,
        show_visual=True,
        show_cls=False,
    )

    joint_prediction = _forward_joint_context_diagnostic(jepa, output, cls_tokens, joint_mask, tokenized_size)
    shuffled_joint_prediction = _forward_joint_context_diagnostic(
        jepa, output, shuffled_cls_tokens, joint_mask, tokenized_size
    )
    cls_only_prediction = _forward_joint_context_diagnostic(jepa, output, cls_tokens, cls_only_mask, tokenized_size)
    shuffled_cls_only_prediction = _forward_joint_context_diagnostic(
        jepa, output, shuffled_cls_tokens, cls_only_mask, tokenized_size
    )
    visual_only_prediction = _forward_joint_context_diagnostic(
        jepa, output, cls_tokens, visual_only_mask, tokenized_size
    )
    target = jepa._masked_target(output.target_mask, output.teacher_output.visual_tokens)
    target = target.repeat(1, joint_context_query_multiplier(cls_prediction_mode), 1)
    loss = partial(compute_jepa_prediction_loss, teacher=target, kind=jepa.config.jepa_loss_kind)
    joint_loss = loss(joint_prediction).item()
    shuffled_joint_loss = loss(shuffled_joint_prediction).item()
    cls_only_loss = loss(cls_only_prediction).item()
    shuffled_cls_only_loss = loss(shuffled_cls_only_prediction).item()
    visual_only_loss = loss(visual_only_prediction).item()
    return {
        "pretrain/validation_cls_aux_loss": cls_only_loss,
        "pretrain/validation_cls_aux_loss_shuffled": shuffled_cls_only_loss,
        "pretrain/validation_cls_aux_shuffle_gap": shuffled_cls_only_loss - cls_only_loss,
        "pretrain/validation_joint_context_loss": joint_loss,
        "pretrain/validation_joint_context_loss_shuffled_cls": shuffled_joint_loss,
        "pretrain/validation_joint_context_cls_shuffle_gap": shuffled_joint_loss - joint_loss,
        "pretrain/validation_visual_only_loss": visual_only_loss,
    }


def _compute_packed_adaln_shuffle_diagnostic(
    jepa: MJEPA,
    output: MJEPAPredictions,
    tokenized_size: tuple[int, int],
) -> dict[str, float]:
    cls_tokens = output.student_output.cls_tokens
    if cls_tokens.shape[1] != 1:
        raise ValueError("packed AdaLN CLS diagnostics require exactly one CLS token")
    shuffled_cls_tokens = torch.roll(cls_tokens, shifts=1, dims=0)
    prediction, _ = jepa.forward_packed_adaln_hard_blind_predictor_heads(
        tokenized_size,
        output.student_output.visual_tokens,
        cls_tokens,
        output.context_mask,
        output.target_mask,
        rope_seed=VALIDATION_DIAGNOSTIC_SEED,
    )
    shuffled_prediction, _ = jepa.forward_packed_adaln_hard_blind_predictor_heads(
        tokenized_size,
        output.student_output.visual_tokens,
        shuffled_cls_tokens,
        output.context_mask,
        output.target_mask,
        rope_seed=VALIDATION_DIAGNOSTIC_SEED,
    )
    target = jepa._masked_target(output.target_mask, output.teacher_output.visual_tokens)
    target_tokens = target.shape[1]
    visual_prediction = prediction[:, :target_tokens]
    blind_prediction = prediction[:, target_tokens:]
    shuffled_blind_prediction = shuffled_prediction[:, target_tokens:]
    loss = partial(compute_jepa_prediction_loss, teacher=target, kind=jepa.config.jepa_loss_kind)
    visual_loss = loss(visual_prediction).item()
    blind_loss = loss(blind_prediction).item()
    shuffled_blind_loss = loss(shuffled_blind_prediction).item()
    return {
        "pretrain/validation_cls_aux_loss": blind_loss,
        "pretrain/validation_cls_aux_loss_shuffled": shuffled_blind_loss,
        "pretrain/validation_cls_aux_shuffle_gap": shuffled_blind_loss - blind_loss,
        "pretrain/validation_visual_only_loss": visual_loss,
    }


def compute_cls_aux_shuffle_diagnostic(jepa: MJEPA, output: MJEPAPredictions) -> dict[str, float]:
    """Measure target prediction dependence on the student's CLS identity."""
    if not MJEPA._has_cls_tokens(output.student_output):
        return {}
    raw_tokenized_size = output.teacher_output.tokenized_size
    if raw_tokenized_size is None or len(raw_tokenized_size) != 2:
        raise ValueError("teacher output must record tokenized_size for CLS diagnostics")
    tokenized_size = cast(tuple[int, int], raw_tokenized_size)
    if jepa.config.cls_prediction_mode in PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES:
        return _compute_packed_adaln_shuffle_diagnostic(jepa, output, tokenized_size)
    if jepa.config.cls_prediction_mode in JOINT_CONTEXT_CLS_PREDICTION_MODES:
        return _compute_joint_context_shuffle_diagnostic(jepa, output, tokenized_size)
    if output.pred_with_cls is None:
        return {}

    cls_tokens = output.student_output.cls_tokens
    shuffled_cls_tokens = torch.roll(cls_tokens, shifts=1, dims=0)
    true_prediction = jepa.forward_cls_predictor(
        tokenized_size,
        cls_tokens,
        output.target_mask,
        rope_seed=VALIDATION_DIAGNOSTIC_SEED,
    )
    shuffled_prediction = jepa.forward_cls_predictor(
        tokenized_size,
        shuffled_cls_tokens,
        output.target_mask,
        rope_seed=VALIDATION_DIAGNOSTIC_SEED,
    )
    target = jepa._masked_target(output.target_mask, output.teacher_output.visual_tokens)
    true_loss = compute_jepa_prediction_loss(
        true_prediction,
        target,
        kind=jepa.config.jepa_loss_kind,
    ).item()
    shuffled_loss = compute_jepa_prediction_loss(
        shuffled_prediction,
        target,
        kind=jepa.config.jepa_loss_kind,
    ).item()
    return {
        "pretrain/validation_cls_aux_loss": true_loss,
        "pretrain/validation_cls_aux_loss_shuffled": shuffled_loss,
        "pretrain/validation_cls_aux_shuffle_gap": shuffled_loss - true_loss,
    }


def compute_visual_target_shuffle_diagnostic(
    jepa: CIFAR10MJEPA,
    output: MJEPAPredictions,
) -> dict[str, float]:
    """Compare masked predictions with their matched and cross-sample targets."""
    target = jepa._masked_target(output.target_mask, output.teacher_output.visual_tokens)
    shuffled_target = torch.roll(target, shifts=1, dims=0)
    position_shuffled_target = torch.roll(target, shifts=1, dims=1)
    broadcast_mean_target = target.mean(dim=1, keepdim=True).expand_as(target)
    true_loss = compute_jepa_prediction_loss(output.pred, target, kind=jepa.config.jepa_loss_kind).item()
    shuffled_loss = compute_jepa_prediction_loss(
        output.pred,
        shuffled_target,
        kind=jepa.config.jepa_loss_kind,
    ).item()
    position_shuffled_loss = compute_jepa_prediction_loss(
        output.pred,
        position_shuffled_target,
        kind=jepa.config.jepa_loss_kind,
    ).item()
    broadcast_mean_loss = compute_jepa_prediction_loss(
        output.pred,
        broadcast_mean_target,
        kind=jepa.config.jepa_loss_kind,
    ).item()
    gap = shuffled_loss - true_loss
    relative_improvement = gap / max(abs(shuffled_loss), LOSS_DENOMINATOR_EPSILON)
    return {
        "pretrain/validation_visual_target_loss": true_loss,
        "pretrain/validation_visual_target_loss_shuffled": shuffled_loss,
        "pretrain/validation_visual_target_shuffle_gap": gap,
        "pretrain/validation_visual_target_relative_improvement": relative_improvement,
        "pretrain/validation_visual_target_loss_position_shuffled": position_shuffled_loss,
        "pretrain/validation_visual_target_position_shuffle_gap": position_shuffled_loss - true_loss,
        "pretrain/validation_visual_target_loss_broadcast_mean": broadcast_mean_loss,
        "pretrain/validation_visual_target_broadcast_mean_gap": broadcast_mean_loss - true_loss,
    }


def _cls_global_target_embeddings(output: MJEPAPredictions) -> tuple[Tensor, Tensor]:
    student_cls_tokens = output.student_output.cls_tokens
    if student_cls_tokens.shape[1] != 1:
        raise ValueError(
            f"CLS global-target loss requires exactly one student CLS token, got {student_cls_tokens.shape[1]}"
        )
    teacher_visual_tokens = output.teacher_output.visual_tokens
    if teacher_visual_tokens.shape[1] == 0:
        raise ValueError("CLS global-target loss requires at least one teacher visual token")
    return student_cls_tokens[:, 0].float(), teacher_visual_tokens.float().mean(dim=1)


def compute_cls_global_target_loss(output: MJEPAPredictions) -> Tensor:
    """Regress one student CLS token directly toward the full teacher visual-token mean."""
    student_cls, teacher_global_target = _cls_global_target_embeddings(output)
    return F.mse_loss(student_cls, teacher_global_target)


def _center_and_normalize_global_embedding(embedding: Tensor) -> Tensor:
    embedding = embedding.float()
    centered = embedding - embedding.mean(dim=-1, keepdim=True)
    return F.normalize(centered, dim=-1)


def _normalized_global_regression_loss(student: Tensor, target: Tensor) -> Tensor:
    student = _center_and_normalize_global_embedding(student)
    target = _center_and_normalize_global_embedding(target.detach())
    return (student - target).square().sum(dim=-1).mean()


def _mean_pairwise_cosine(normalized_embeddings: Tensor) -> float:
    batch_size = normalized_embeddings.shape[0]
    if batch_size < 2:
        return 1.0
    off_diagonal_sum = normalized_embeddings.sum(dim=0).square().sum() - normalized_embeddings.square().sum()
    return off_diagonal_sum.div(batch_size * (batch_size - 1)).clamp(-1.0, 1.0).item()


def _cls_global_target_views(
    jepa: CIFAR10MJEPA,
    output: MJEPAPredictions,
) -> tuple[Tensor, Tensor | None, Tensor]:
    student_cls, raw_mean_target = _cls_global_target_embeddings(output)
    pooling = jepa.cls_global_target_pooling
    if pooling == RAW_MEAN_CLS_GLOBAL_TARGET_POOLING:
        return student_cls, None, raw_mean_target.detach()
    if pooling == CENTERED_NORMALIZED_MEAN_CLS_GLOBAL_TARGET_POOLING:
        student_pooled = output.student_output.visual_tokens.float().mean(dim=1)
        return student_cls, student_pooled, raw_mean_target.detach()
    if pooling == CENTERED_NORMALIZED_EMA_ATTENTION_CLS_GLOBAL_TARGET_POOLING:
        poolers = jepa.cls_global_target_poolers
        if poolers is None:
            raise RuntimeError("EMA attention global-target pooling is not initialized")
        student_pooled = poolers.online(output.student_output.visual_tokens.float())
        poolers.target.eval()
        with torch.no_grad():
            teacher_target = poolers.target(output.teacher_output.visual_tokens.float())
        return student_cls, student_pooled, teacher_target.detach()
    raise ValueError(f"Unsupported CLS global-target pooling mode: {pooling!r}")


def compute_cls_global_target_objective(
    jepa: CIFAR10MJEPA,
    output: MJEPAPredictions,
) -> CLSGlobalTargetLosses:
    """Build a stopped full-teacher target and its direct CLS and visible-pool losses."""
    student_cls, student_pooled, teacher_target = _cls_global_target_views(jepa, output)
    if jepa.cls_global_target_pooling == RAW_MEAN_CLS_GLOBAL_TARGET_POOLING:
        return CLSGlobalTargetLosses(
            cls_loss=F.mse_loss(student_cls, teacher_target),
            pool_consistency_loss=None,
            teacher_target=teacher_target,
        )
    teacher_target = _center_and_normalize_global_embedding(teacher_target).detach()
    cls_loss = _normalized_global_regression_loss(student_cls, teacher_target)
    pool_consistency_loss = (
        _normalized_global_regression_loss(student_pooled, teacher_target) if student_pooled is not None else None
    )
    return CLSGlobalTargetLosses(
        cls_loss=cls_loss,
        pool_consistency_loss=pool_consistency_loss,
        teacher_target=teacher_target,
    )


def compute_cls_global_target_diagnostic(
    output: MJEPAPredictions,
    jepa: CIFAR10MJEPA | None = None,
) -> dict[str, float]:
    """Measure global-target loss before and after cyclically shuffling CLS identity."""
    student_cls, raw_teacher_target = _cls_global_target_embeddings(output)
    if jepa is None or jepa.cls_global_target_pooling == RAW_MEAN_CLS_GLOBAL_TARGET_POOLING:
        teacher_global_target = raw_teacher_target
        true_loss = F.mse_loss(student_cls, teacher_global_target).item()
        shuffled_loss = F.mse_loss(torch.roll(student_cls, shifts=1, dims=0), teacher_global_target).item()
        return {
            "pretrain/validation_cls_global_loss": true_loss,
            "pretrain/validation_cls_global_loss_shuffled": shuffled_loss,
            "pretrain/validation_cls_global_shuffle_gap": shuffled_loss - true_loss,
            "pretrain/validation_cls_global_student_norm": student_cls.norm(dim=-1).mean().item(),
            "pretrain/validation_cls_global_teacher_norm": teacher_global_target.norm(dim=-1).mean().item(),
        }
    losses = compute_cls_global_target_objective(jepa, output)
    teacher_global_target = losses.teacher_target
    normalized_student_cls = _center_and_normalize_global_embedding(student_cls)
    shuffled_student_cls = torch.roll(student_cls, shifts=1, dims=0)
    true_loss = losses.cls_loss.item()
    shuffled_loss = _normalized_global_regression_loss(shuffled_student_cls, teacher_global_target).item()
    metrics = {
        "pretrain/validation_cls_global_loss": true_loss,
        "pretrain/validation_cls_global_loss_shuffled": shuffled_loss,
        "pretrain/validation_cls_global_shuffle_gap": shuffled_loss - true_loss,
        "pretrain/validation_cls_global_student_norm": normalized_student_cls.norm(dim=-1).mean().item(),
        "pretrain/validation_cls_global_teacher_norm": teacher_global_target.norm(dim=-1).mean().item(),
        "pretrain/validation_cls_global_teacher_batch_std": teacher_global_target.std(dim=0).mean().item(),
        "pretrain/validation_cls_global_teacher_mean_pairwise_cosine": _mean_pairwise_cosine(teacher_global_target),
    }
    if losses.pool_consistency_loss is not None:
        metrics["pretrain/validation_cls_global_pool_consistency_loss"] = losses.pool_consistency_loss.item()
    if (poolers := jepa.cls_global_target_poolers) is not None:
        target_weights = poolers.target.forward_weights(output.teacher_output.visual_tokens.float())
        num_tokens = target_weights.shape[-1]
        normalized_entropy = (
            -(target_weights * target_weights.clamp_min(torch.finfo(target_weights.dtype).tiny).log())
            .sum(dim=-1)
            .div(math.log(num_tokens))
            .mean()
            if num_tokens > 1
            else target_weights.new_zeros(())
        )
        metrics["pretrain/validation_cls_global_target_attention_normalized_entropy"] = normalized_entropy.item()
        metrics["pretrain/validation_cls_global_target_attention_max_weight"] = (
            target_weights.max(dim=-1).values.mean().item()
        )
    return metrics


def run_optimizer_step(
    optimizer: OptimizerLike,
    scheduler: SchedulerLike,
    step: int,
    total_steps: int,
    max_grad_norm: float | None = None,
    update_teacher: Callable[[], None] | None = None,
) -> OptimizerStepResult:
    total_grad_norm = clip_optimizer_grad_norm_(optimizer, max_grad_norm)
    if step < total_steps:
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()
    if update_teacher is not None:
        update_teacher()
    return OptimizerStepResult(
        next_step=step + 1,
        grad_clip_triggered=did_gradient_clip(total_grad_norm, max_grad_norm),
    )


def train(
    jepa: MJEPA | DDP,
    train_dataloader_fn: DataLoaderFn,
    val_dataloader_fn: DataLoaderFn,
    optimizer: OptimizerLike,
    scheduler: SchedulerLike,
    trainer_config: TrainerConfig,
    test_dataloader_fn: DataLoaderFn | None = None,
    last_epoch: int = -1,
    initial_step: int | None = None,
    elapsed_seconds_offset: float = 0.0,
    wandb_run_id: str | None = None,
    output_dir: Path | None = None,
    max_grad_norm: float | None = None,
    cls_global_target_loss_weight: float = DEFAULT_CLS_GLOBAL_TARGET_LOSS_WEIGHT,
    cls_global_pool_consistency_loss_weight: float = DEFAULT_CLS_GLOBAL_POOL_CONSISTENCY_LOSS_WEIGHT,
    progress_callback: ProgressCallback | None = None,
    first_cycle_callback: FirstCycleCallback | None = None,
) -> None:
    training_started_at = perf_counter()
    # Module setup
    log_dir = output_dir if output_dir is not None else (Path(wandb.run.dir) if wandb.run is not None else None)
    unwrapped_jepa = jepa.module if isinstance(jepa, DDP) else jepa
    assert isinstance(unwrapped_jepa, CIFAR10MJEPA)
    if cls_global_target_loss_weight < 0:
        raise ValueError("CLS global-target loss weight must be non-negative")
    if cls_global_pool_consistency_loss_weight < 0:
        raise ValueError("CLS global pool-consistency loss weight must be non-negative")
    if cls_global_target_loss_weight > 0 or cls_global_pool_consistency_loss_weight > 0:
        if unwrapped_jepa.student.config.num_cls_tokens != 1:
            raise ValueError("CLS global-target loss requires exactly one student CLS token")
        supported_prediction_modes = {
            ADALN_BLIND_CLS_PREDICTION_MODE,
            *PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES,
        }
        if unwrapped_jepa.config.cls_prediction_mode not in supported_prediction_modes:
            raise ValueError("CLS global-target loss requires a visually blinded AdaLN predictor mode")
    if (
        cls_global_pool_consistency_loss_weight > 0
        and unwrapped_jepa.cls_global_target_pooling == RAW_MEAN_CLS_GLOBAL_TARGET_POOLING
    ):
        raise ValueError("CLS global pool-consistency loss requires centered normalized pooling")
    if (
        unwrapped_jepa.cls_global_target_pooling == CENTERED_NORMALIZED_EMA_ATTENTION_CLS_GLOBAL_TARGET_POOLING
        and cls_global_pool_consistency_loss_weight <= 0
    ):
        raise ValueError("EMA attention global-target pooling requires a positive pool-consistency loss weight")
    optimizer.zero_grad()

    # DataLoader setup
    train_dataloader = train_dataloader_fn(unwrapped_jepa.img_size, trainer_config.batch_size)
    val_dataloader = val_dataloader_fn(unwrapped_jepa.img_size, trainer_config.batch_size)
    jepa_scale = unwrapped_jepa.config.scale

    accumulate_grad_batches = trainer_config.accumulate_grad_batches
    microbatch = (last_epoch + 1) * len(train_dataloader)
    inferred_step = microbatch // accumulate_grad_batches
    step = inferred_step if initial_step is None else initial_step
    total_steps = calculate_total_steps(train_dataloader, trainer_config.num_epochs, accumulate_grad_batches)
    rank_zero_info(f"Training for {trainer_config.num_epochs} epochs = {total_steps} steps")
    rank_zero_info(
        f"Batch size: {trainer_config.batch_size}, Microbatch accumulation: {trainer_config.accumulate_grad_batches}"
    )
    first_cycle_reported = False

    def active_seconds() -> float:
        return elapsed_seconds_offset + perf_counter() - training_started_at

    if is_rank_zero() and progress_callback is not None:
        progress_callback("training", max(last_epoch + 1, 0), step, active_seconds())

    # Metric setup
    train_loss = tm.RunningMean(window=WINDOW).cuda()
    train_loss_jepa = tm.RunningMean(window=WINDOW).cuda()
    train_loss_jepa_cls = tm.RunningMean(window=WINDOW).cuda()
    train_loss_cls_global = tm.RunningMean(window=WINDOW).cuda()
    train_loss_cls_global_pool = tm.RunningMean(window=WINDOW).cuda()
    train_loss_sigreg = tm.RunningMean(window=WINDOW).cuda()
    train_loss_invariance = tm.RunningMean(window=WINDOW).cuda()
    train_loss_gram = tm.RunningMean(window=WINDOW).cuda()
    has_jepa_loss_cls = False
    has_sigreg_loss = False
    has_invariance_loss = False
    has_gram_loss = False
    train_acc = Running(tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES), window=WINDOW).cuda()
    train_grad_clip_trigger_pct = tm.MeanMetric().cuda() if max_grad_norm is not None else None
    val_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).cuda()
    train_cpa = CLSPatchAlignmentMetric().cuda() if unwrapped_jepa.student.config.num_cls_tokens > 0 else None
    val_cpa = CLSPatchAlignmentMetric().cuda() if unwrapped_jepa.student.config.num_cls_tokens > 0 else None
    embedding_dim = unwrapped_jepa.student.config.hidden_size
    val_target_cls_collapse = (
        EmbeddingCollapseMetric(embedding_dim).cuda() if unwrapped_jepa.student.config.num_cls_tokens > 0 else None
    )
    val_target_patch_collapse = EmbeddingCollapseMetric(embedding_dim).cuda()
    val_target_patch_diversity = PatchTokenDiversityMetric(embedding_dim).cuda()
    val_projected_target_collapse = (
        EmbeddingCollapseMetric(unwrapped_jepa.config.sigreg_projector_dims[-1]).cuda()
        if unwrapped_jepa.config.sigreg_projector_dims
        else None
    )

    img: Tensor
    label: Tensor
    for epoch in range(last_epoch + 1, trainer_config.num_epochs):
        # Update training resolution / batch_size / accumulate_grad_batches if necessary
        if trainer_config.is_size_change_epoch(epoch):
            size_config = trainer_config.sizes[epoch]
            train_dataloader, val_dataloader, accumulate_grad_batches = size_change(
                size_config,
                trainer_config.batch_size,
                accumulate_grad_batches,
                train_dataloader_fn,
                val_dataloader_fn,
            )
            jepa_scale = scale_change(unwrapped_jepa.img_size, size_config, unwrapped_jepa.config.scale)
            rank_zero_info(
                f"Changing size to {size_config.size} and batch size to {size_config.batch_size} "
                f"(accumulate grad batches: {accumulate_grad_batches}, jepa scale: {jepa_scale})"
            )

        # Update sampler epoch for proper shuffling in DDP
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        jepa.train()
        desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
        pbar = tqdm(train_dataloader, desc=desc, disable=not is_rank_zero(), leave=False)
        for img, label in pbar:
            B = img.shape[0]
            img = img.cuda()
            img, additional_views = split_training_views(img)
            label = label.cuda()
            should_step = should_step_optimizer(microbatch + 1, accumulate_grad_batches)
            with get_gradient_sync_context(jepa.no_sync if isinstance(jepa, DDP) else None, should_step):
                output = jepa(img, jepa_scale, epoch, additional_views=additional_views)
                assert isinstance(output, MJEPAPredictions)
                ssl_losses = unwrapped_jepa.compute_losses(output, step, epoch)
                train_loss_jepa.update(ssl_losses.jepa_loss)

                jepa_loss_cls = getattr(ssl_losses, "jepa_loss_cls", None)
                if jepa_loss_cls is not None:
                    train_loss_jepa_cls.update(jepa_loss_cls)
                    has_jepa_loss_cls = True

                sigreg_loss = getattr(ssl_losses, "sigreg_loss", None)
                if sigreg_loss is not None:
                    train_loss_sigreg.update(sigreg_loss)
                    has_sigreg_loss = True

                invariance_loss = getattr(ssl_losses, "invariance_loss", None)
                if isinstance(invariance_loss, Tensor):
                    train_loss_invariance.update(invariance_loss)
                    has_invariance_loss = True

                gram_loss = getattr(ssl_losses, "gram_loss", None)
                if gram_loss is not None:
                    train_loss_gram.update(gram_loss)
                    has_gram_loss = True

                ssl_loss = ssl_losses.reduce()
                if cls_global_target_loss_weight > 0 or cls_global_pool_consistency_loss_weight > 0:
                    global_target_losses = compute_cls_global_target_objective(unwrapped_jepa, output)
                    if cls_global_target_loss_weight > 0:
                        train_loss_cls_global.update(global_target_losses.cls_loss)
                        ssl_loss = ssl_loss + global_target_losses.cls_loss * cls_global_target_loss_weight
                    if cls_global_pool_consistency_loss_weight > 0:
                        if global_target_losses.pool_consistency_loss is None:
                            raise RuntimeError("Configured global pool-consistency loss is unavailable")
                        train_loss_cls_global_pool.update(global_target_losses.pool_consistency_loss)
                        ssl_loss = (
                            ssl_loss
                            + global_target_losses.pool_consistency_loss * cls_global_pool_consistency_loss_weight
                        )

                # Compute linear probe loss
                probe_pred = output.probes["cls"]
                probe_loss = F.cross_entropy(probe_pred, label)

                # Combine losses
                loss = ssl_loss + probe_loss
                train_loss.update(loss)

                with torch.no_grad():
                    train_acc.update(probe_pred, label)
                    update_cls_patch_alignment_metric(train_cpa, output.teacher_output)

                # Backward
                assert not loss.isnan()
                loss.backward()
            unwrapped_jepa.assert_student_params_have_grad(microbatch)
            unwrapped_jepa.assert_predictor_params_have_grad(microbatch)
            microbatch += 1
            should_log_train_metrics = should_step and (step + 1) % LOG_INTERVAL == 0
            grad_norm_stats = None
            if should_log_train_metrics and is_rank_zero():
                grad_norm_stats = get_gradient_norm_stats(unwrapped_jepa.parameters())

            # Optimizer update and teacher update
            if should_step:
                update_teacher = (
                    partial(unwrapped_jepa.update_teacher, step, total_steps)
                    if unwrapped_jepa.teacher is not None
                    else None
                )
                optimizer_step_result = run_optimizer_step(
                    optimizer,
                    scheduler,
                    step,
                    total_steps,
                    max_grad_norm=max_grad_norm,
                    update_teacher=update_teacher,
                )
                step = optimizer_step_result.next_step
                if train_grad_clip_trigger_pct is not None:
                    train_grad_clip_trigger_pct.update(float(optimizer_step_result.grad_clip_triggered))

            desc = format_pbar_description(step, microbatch, epoch, loss=train_loss, acc=train_acc)
            pbar.set_description(desc)

            # Log to wandb
            if step % LOG_INTERVAL == 0 and microbatch % accumulate_grad_batches == 0:
                log_dict = {
                    "pretrain/loss": train_loss.compute().item(),
                    "pretrain/loss_jepa": train_loss_jepa.compute().item(),
                    "probe/train_accuracy": train_acc.compute().item(),
                    "pretrain/lr": get_scheduler_last_lr(scheduler),
                    "convergence/active_seconds": elapsed_seconds_offset + perf_counter() - training_started_at,
                }
                if grad_norm_stats is not None:
                    grad_norm_mean, grad_norm_max = grad_norm_stats
                    log_dict["pretrain/grad_norm_mean"] = grad_norm_mean
                    log_dict["pretrain/grad_norm_max"] = grad_norm_max
                if train_grad_clip_trigger_pct is not None:
                    log_dict[GRAD_CLIP_TRIGGER_PCT_KEY] = compute_and_reset_mean_percentage(train_grad_clip_trigger_pct)
                if has_jepa_loss_cls:
                    log_dict["pretrain/loss_jepa_cls"] = train_loss_jepa_cls.compute().item()
                if cls_global_target_loss_weight > 0:
                    cls_global_target_loss_value = train_loss_cls_global.compute().item()
                    log_dict["pretrain/loss_cls_global"] = cls_global_target_loss_value
                    log_dict["pretrain/loss_cls_global_weighted"] = (
                        cls_global_target_loss_value * cls_global_target_loss_weight
                    )
                if cls_global_pool_consistency_loss_weight > 0:
                    pool_consistency_loss_value = train_loss_cls_global_pool.compute().item()
                    log_dict["pretrain/loss_cls_global_pool_consistency"] = pool_consistency_loss_value
                    log_dict["pretrain/loss_cls_global_pool_consistency_weighted"] = (
                        pool_consistency_loss_value * cls_global_pool_consistency_loss_weight
                    )
                if has_sigreg_loss:
                    log_dict["pretrain/loss_sigreg"] = train_loss_sigreg.compute().item()
                if has_invariance_loss:
                    log_dict["pretrain/loss_invariance"] = train_loss_invariance.compute().item()
                if has_gram_loss:
                    log_dict["pretrain/loss_gram"] = train_loss_gram.compute().item()
                if train_cpa is not None:
                    log_dict.update(compute_and_reset_cpa_metrics(train_cpa, prefix="pretrain/train"))
                if is_rank_zero():
                    wandb.log(log_dict, step=step)
                    if progress_callback is not None:
                        progress_callback("training", epoch, step, active_seconds())

        # Validation
        validation_completed = False
        pbar.close()
        unwrapped_jepa.assert_student_params_synced()
        unwrapped_jepa.assert_predictor_params_synced()
        if val_dataloader is not None and (epoch + 1) % trainer_config.check_val_every_n_epoch == 0:
            if is_rank_zero() and progress_callback is not None:
                progress_callback("validation", epoch, step, active_seconds())
            jepa.eval()
            val_acc.reset()
            if val_cpa is not None:
                val_cpa.reset()
            if val_target_cls_collapse is not None:
                val_target_cls_collapse.reset()
            val_target_patch_collapse.reset()
            val_target_patch_diversity.reset()
            if val_projected_target_collapse is not None:
                val_projected_target_collapse.reset()

            cls_aux_diagnostics: dict[str, float] = {}
            cls_global_target_diagnostics: dict[str, float] = {}
            visual_target_diagnostics: dict[str, float] = {}
            for batch_index, (img, label) in enumerate(
                tqdm(val_dataloader, desc="Validating: ", disable=not is_rank_zero(), leave=False)
            ):
                B = img.shape[0]
                img = img.cuda()
                label = label.cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if batch_index == 0:
                        with torch.random.fork_rng(devices=[img.device]):
                            torch.manual_seed(VALIDATION_DIAGNOSTIC_SEED)
                            diagnostic_output = unwrapped_jepa(img, jepa_scale, epoch)
                        output = diagnostic_output.teacher_output
                        probe_pred = diagnostic_output.probes["cls"].view(B, -1)
                        cls_aux_diagnostics = compute_cls_aux_shuffle_diagnostic(
                            unwrapped_jepa,
                            diagnostic_output,
                        )
                        visual_target_diagnostics = compute_visual_target_shuffle_diagnostic(
                            unwrapped_jepa,
                            diagnostic_output,
                        )
                        if diagnostic_output.student_output.num_cls_tokens == 1:
                            cls_global_target_diagnostics = compute_cls_global_target_diagnostic(
                                diagnostic_output,
                                unwrapped_jepa,
                            )
                    else:
                        output = unwrapped_jepa.forward_target(img)
                        probe_pred = unwrapped_jepa.forward_probe(output)["cls"].view(B, -1)
                    val_acc.update(probe_pred, label)
                    update_cls_patch_alignment_metric(val_cpa, output)
                    if val_target_cls_collapse is not None:
                        target_cls = output.cls_tokens[:, 0]
                        val_target_cls_collapse.update(target_cls)
                        if val_projected_target_collapse is not None:
                            val_projected_target_collapse.update(unwrapped_jepa.project_sigreg_embeddings(target_cls))
                    val_target_patch_collapse.update(output.visual_tokens.mean(dim=1))
                    val_target_patch_diversity.update(output.visual_tokens)

            # Validation epoch end
            val_acc_value = val_acc.compute()
            rank_zero_info(f"Epoch: {epoch}, Val Acc: {val_acc_value:.4f}")

            # Log validation to wandb
            log_dict = {
                "probe/validation_accuracy": val_acc_value.item(),
                "probe/validation_epoch": epoch,
                "convergence/active_seconds": elapsed_seconds_offset + perf_counter() - training_started_at,
            }
            if val_cpa is not None:
                log_dict.update(compute_and_reset_cpa_metrics(val_cpa, prefix="pretrain/validation"))
            if val_target_cls_collapse is not None:
                log_dict.update(
                    compute_and_reset_collapse_metrics(
                        val_target_cls_collapse,
                        prefix="pretrain/collapse/target_cls",
                    )
                )
            log_dict.update(
                compute_and_reset_collapse_metrics(
                    val_target_patch_collapse,
                    prefix="pretrain/collapse/target_patch_mean",
                )
            )
            log_dict.update(
                compute_and_reset_patch_token_diversity_metrics(
                    val_target_patch_diversity,
                    prefix="pretrain/diversity/target_patch",
                )
            )
            if val_projected_target_collapse is not None:
                log_dict.update(
                    compute_and_reset_collapse_metrics(
                        val_projected_target_collapse,
                        prefix="pretrain/collapse/projected_target_cls",
                    )
                )
            log_dict.update(cls_aux_diagnostics)
            log_dict.update(cls_global_target_diagnostics)
            log_dict.update(visual_target_diagnostics)

            # Add histogram logging
            if is_rank_zero():
                wandb.log(log_dict, step=step)
                append_metric_record(log_dir, step, log_dict)
                validation_completed = True

        # Save checkpoint
        if is_rank_zero() and log_dir:
            if progress_callback is not None:
                progress_callback("checkpointing", epoch, step, active_seconds())
            save_checkpoint(
                path=log_dir / "checkpoint.pt",
                backbone=unwrapped_jepa.student,
                predictor=unwrapped_jepa.predictor,
                teacher=unwrapped_jepa.teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
                elapsed_seconds=elapsed_seconds_offset + perf_counter() - training_started_at,
                wandb_run_id=wandb_run_id,
            )
            save_safetensors_atomic(
                log_dir / "backbone.safetensors",
                {k: v for k, v in unwrapped_jepa.student.state_dict().items() if isinstance(v, torch.Tensor)},
            )
            first_cycle_reported = report_checkpoint_lifecycle(
                progress_callback=progress_callback,
                first_cycle_callback=first_cycle_callback,
                validation_completed=validation_completed,
                first_cycle_reported=first_cycle_reported,
                epoch=epoch,
                optimizer_step=step,
                active_seconds=active_seconds(),
            )

    # Save final checkpoint
    if is_rank_zero() and log_dir:
        save_safetensors_atomic(
            log_dir / "backbone.safetensors",
            {k: v for k, v in unwrapped_jepa.student.state_dict().items() if isinstance(v, torch.Tensor)},
        )

    if test_dataloader_fn is not None:
        test_dataloader = test_dataloader_fn(unwrapped_jepa.img_size, trainer_config.batch_size)
        test_acc = tm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).cuda()
        jepa.eval()
        for img, label in tqdm(test_dataloader, desc="Testing probe: ", disable=not is_rank_zero(), leave=False):
            batch_size = img.shape[0]
            img = img.cuda()
            label = label.cuda()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                target_output = unwrapped_jepa.forward_target(img)
                probe_pred = unwrapped_jepa.forward_probe(target_output)["cls"].view(batch_size, -1)
                test_acc.update(probe_pred, label)
        if is_rank_zero():
            test_log_dict = {
                "probe/test_accuracy": test_acc.compute().item(),
                "convergence/active_seconds": elapsed_seconds_offset + perf_counter() - training_started_at,
            }
            wandb.log(test_log_dict, step=step)
            append_metric_record(log_dir, step, test_log_dict)
