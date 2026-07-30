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


def compute_and_reset_collapse_metrics(metric: EmbeddingCollapseMetric, prefix: str) -> dict[str, float]:
    """Compute a collapse metric, reset its state, and prefix scalar keys."""
    values = metric.compute()
    metric.reset()
    return {f"{prefix}/{key}": value.item() for key, value in values.items()}
