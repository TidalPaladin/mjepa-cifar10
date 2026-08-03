import torch
from torch import Tensor
from torchmetrics import Metric


NUMERICAL_EPSILON = torch.finfo(torch.float64).eps


class EmbeddingCollapseMetric(Metric):
    """Accumulate distributed embedding statistics needed for collapse checks."""

    full_state_update = False
    row_count: Tensor
    finite_row_count: Tensor
    feature_sum: Tensor
    feature_outer_sum: Tensor
    normalized_feature_sum: Tensor
    normalized_squared_norm_sum: Tensor
    feature_norm_sum: Tensor

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.add_state("row_count", default=torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("finite_row_count", default=torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("feature_sum", default=torch.zeros(embedding_dim, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state(
            "feature_outer_sum",
            default=torch.zeros(embedding_dim, embedding_dim, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "normalized_feature_sum",
            default=torch.zeros(embedding_dim, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "normalized_squared_norm_sum",
            default=torch.zeros((), dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("feature_norm_sum", default=torch.zeros((), dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, embeddings: Tensor) -> None:
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"expected embeddings with dimension {self.embedding_dim}, got {embeddings.shape[-1]}")
        rows = embeddings.detach().reshape(-1, self.embedding_dim).to(dtype=torch.float64)
        self.row_count += rows.shape[0]
        finite_rows = rows[torch.isfinite(rows).all(dim=-1)]
        if finite_rows.shape[0] == 0:
            return

        self.finite_row_count += finite_rows.shape[0]
        self.feature_sum += finite_rows.sum(dim=0)
        self.feature_outer_sum += finite_rows.T @ finite_rows
        norms = finite_rows.norm(dim=-1)
        normalized = finite_rows / norms.clamp_min(NUMERICAL_EPSILON).unsqueeze(-1)
        normalized = torch.where(norms.unsqueeze(-1) > NUMERICAL_EPSILON, normalized, 0.0)
        self.normalized_feature_sum += normalized.sum(dim=0)
        self.normalized_squared_norm_sum += normalized.square().sum()
        self.feature_norm_sum += norms.sum()

    def compute(self) -> dict[str, Tensor]:
        total_rows = self.row_count.clamp_min(1)
        finite_rows = self.finite_row_count.clamp_min(1)
        finite_fraction = self.finite_row_count / total_rows
        mean = self.feature_sum / finite_rows
        covariance = self.feature_outer_sum / finite_rows - torch.outer(mean, mean)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        total_variance = eigenvalues.sum()
        variances = covariance.diagonal().clamp_min(0)
        standard_deviations = variances.sqrt()

        if total_variance <= NUMERICAL_EPSILON:
            effective_rank_fraction = total_variance.new_zeros(())
            top_eigenvalue_fraction = total_variance.new_ones(())
        else:
            eigenvalue_probabilities = eigenvalues / total_variance
            nonzero_probabilities = eigenvalue_probabilities[eigenvalue_probabilities > NUMERICAL_EPSILON]
            effective_rank = torch.exp(-(nonzero_probabilities * nonzero_probabilities.log()).sum())
            effective_rank_fraction = effective_rank / self.embedding_dim
            top_eigenvalue_fraction = eigenvalues[-1] / total_variance

        if self.finite_row_count < 2:
            mean_pairwise_cosine = total_variance.new_zeros(())
        else:
            pairwise_sum = self.normalized_feature_sum.square().sum() - self.normalized_squared_norm_sum
            pair_count = self.finite_row_count * (self.finite_row_count - 1)
            mean_pairwise_cosine = pairwise_sum / pair_count

        return {
            "std_mean": standard_deviations.mean(),
            "std_min": standard_deviations.min(),
            "effective_rank_fraction": effective_rank_fraction,
            "top_eigenvalue_fraction": top_eigenvalue_fraction,
            "mean_pairwise_cosine": mean_pairwise_cosine,
            "mean_norm": self.feature_norm_sum / finite_rows,
            "finite_fraction": finite_fraction,
        }


class PatchTokenDiversityMetric(Metric):
    """Accumulate within-image patch diversity and centered spatial-rank statistics."""

    full_state_update = False
    image_count: Tensor
    finite_image_count: Tensor
    patch_pair_count: Tensor
    patch_pair_cosine_sum: Tensor
    total_patch_energy_sum: Tensor
    centered_patch_energy_sum: Tensor
    centered_patch_outer_sum: Tensor
    centered_patch_count: Tensor

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.add_state("image_count", default=torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("finite_image_count", default=torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("patch_pair_count", default=torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("patch_pair_cosine_sum", default=torch.zeros((), dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total_patch_energy_sum", default=torch.zeros((), dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("centered_patch_energy_sum", default=torch.zeros((), dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state(
            "centered_patch_outer_sum",
            default=torch.zeros(embedding_dim, embedding_dim, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("centered_patch_count", default=torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, patch_tokens: Tensor) -> None:
        if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected patch tokens with shape [images, patches, {self.embedding_dim}], "
                f"got {tuple(patch_tokens.shape)}"
            )
        if patch_tokens.shape[1] < 2:
            raise ValueError("patch diversity requires at least two patches per image")

        patches = patch_tokens.detach().to(dtype=torch.float64)
        self.image_count += patches.shape[0]
        finite_patches = patches[torch.isfinite(patches).all(dim=(1, 2))]
        if finite_patches.shape[0] == 0:
            return

        self.finite_image_count += finite_patches.shape[0]
        patch_count = finite_patches.shape[1]
        norms = finite_patches.norm(dim=-1)
        normalized = finite_patches / norms.clamp_min(NUMERICAL_EPSILON).unsqueeze(-1)
        normalized = torch.where(norms.unsqueeze(-1) > NUMERICAL_EPSILON, normalized, 0.0)
        normalized_sum = normalized.sum(dim=1)
        diagonal_sum = normalized.square().sum(dim=(1, 2))
        self.patch_pair_cosine_sum += (normalized_sum.square().sum(dim=-1) - diagonal_sum).sum()
        self.patch_pair_count += finite_patches.shape[0] * patch_count * (patch_count - 1)

        centered = finite_patches - finite_patches.mean(dim=1, keepdim=True)
        centered_rows = centered.reshape(-1, self.embedding_dim)
        self.total_patch_energy_sum += finite_patches.square().sum()
        self.centered_patch_energy_sum += centered_rows.square().sum()
        self.centered_patch_outer_sum += centered_rows.T @ centered_rows
        self.centered_patch_count += centered_rows.shape[0]

    def compute(self) -> dict[str, Tensor]:
        total_images = self.image_count.clamp_min(1)
        finite_fraction = self.finite_image_count / total_images
        pairwise_cosine = self.patch_pair_cosine_sum / self.patch_pair_count.clamp_min(1)
        centered_energy_ratio = self.centered_patch_energy_sum / self.total_patch_energy_sum.clamp_min(
            NUMERICAL_EPSILON
        )

        covariance = self.centered_patch_outer_sum / self.centered_patch_count.clamp_min(1)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        total_variance = eigenvalues.sum()
        if total_variance <= NUMERICAL_EPSILON:
            effective_rank_fraction = total_variance.new_zeros(())
        else:
            probabilities = eigenvalues / total_variance
            nonzero_probabilities = probabilities[probabilities > NUMERICAL_EPSILON]
            effective_rank = torch.exp(-(nonzero_probabilities * nonzero_probabilities.log()).sum())
            effective_rank_fraction = effective_rank / self.embedding_dim

        return {
            "mean_within_image_pairwise_cosine": pairwise_cosine,
            "centered_patch_energy_ratio": centered_energy_ratio,
            "centered_patch_effective_rank_fraction": effective_rank_fraction,
            "finite_image_fraction": finite_fraction,
        }


def compute_and_reset_collapse_metrics(metric: EmbeddingCollapseMetric, prefix: str) -> dict[str, float]:
    """Compute a collapse metric, reset its state, and prefix scalar keys."""
    values = metric.compute()
    metric.reset()
    return {f"{prefix}/{key}": value.item() for key, value in values.items()}


def compute_and_reset_patch_token_diversity_metrics(
    metric: PatchTokenDiversityMetric,
    prefix: str,
) -> dict[str, float]:
    """Compute patch diversity metrics, reset their state, and prefix scalar keys."""
    values = metric.compute()
    metric.reset()
    return {f"{prefix}/{key}": value.item() for key, value in values.items()}
