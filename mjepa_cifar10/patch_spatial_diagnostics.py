from dataclasses import dataclass
from math import prod
from typing import Final, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


PCA_COMPONENTS: Final[int] = 3
PCA_LOWER_QUANTILE: Final[float] = 0.01
PCA_UPPER_QUANTILE: Final[float] = 0.99


@dataclass(frozen=True)
class PCARGBBasis:
    feature_mean: Tensor
    components: Tensor
    lower_bounds: Tensor
    upper_bounds: Tensor
    explained_variance_ratio: Tensor


def center_visual_tokens(tokens: Tensor) -> Tensor:
    """Remove each image's patch-token mean to isolate within-image variation."""
    if tokens.ndim != 3:
        raise ValueError(f"expected visual tokens with shape [batch, patches, features], got {tuple(tokens.shape)}")
    return tokens - tokens.mean(dim=1, keepdim=True)


def manhattan_distance_matrix(tokenized_size: Sequence[int], *, device: torch.device | None = None) -> Tensor:
    if len(tokenized_size) != 2 or any(int(size) <= 0 for size in tokenized_size):
        raise ValueError(f"tokenized_size must contain two positive dimensions, got {tuple(tokenized_size)}")
    height, width = (int(size) for size in tokenized_size)
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coordinates = torch.stack((rows.flatten(), columns.flatten()), dim=1)
    return (coordinates[:, None, :] - coordinates[None, :, :]).abs().sum(dim=-1)


def distance_binned_cosine_sums(
    tokens: Tensor,
    tokenized_size: Sequence[int],
    *,
    center_within_image: bool,
) -> tuple[Tensor, Tensor]:
    """Return cosine-similarity sums and counts for unique pairs at each Manhattan distance."""
    if tokens.ndim != 3:
        raise ValueError(f"expected visual tokens with shape [batch, patches, features], got {tuple(tokens.shape)}")
    if tokens.shape[1] != prod(int(size) for size in tokenized_size):
        raise ValueError("visual-token count does not match tokenized_size")

    selected_tokens = center_visual_tokens(tokens) if center_within_image else tokens
    normalized_tokens = F.normalize(selected_tokens.float(), dim=-1)
    similarities = normalized_tokens @ normalized_tokens.transpose(-1, -2)
    distances = manhattan_distance_matrix(tokenized_size, device=tokens.device)
    maximum_distance = int(distances.max().item())
    upper_triangle = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    sums = torch.zeros(maximum_distance, dtype=torch.float64, device=tokens.device)
    counts = torch.zeros(maximum_distance, dtype=torch.int64, device=tokens.device)
    for distance in range(1, maximum_distance + 1):
        mask = upper_triangle & distances.eq(distance)
        sums[distance - 1] = similarities[:, mask].double().sum()
        counts[distance - 1] = mask.sum() * tokens.shape[0]
    return sums, counts


def fit_centered_pca_rgb(tokens: Tensor) -> PCARGBBasis:
    """Fit a deterministic three-component PCA color basis to within-image patch variation."""
    if tokens.ndim != 3 or tokens.shape[-1] < PCA_COMPONENTS:
        raise ValueError("PCA requires [batch, patches, features] with at least three features")
    flattened = center_visual_tokens(tokens).float().flatten(0, 1)
    feature_mean = flattened.mean(dim=0)
    centered = flattened - feature_mean
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    selected_indices = torch.arange(eigenvalues.numel() - 1, eigenvalues.numel() - PCA_COMPONENTS - 1, -1)
    selected_values = eigenvalues[selected_indices]
    components = eigenvectors[:, selected_indices]

    largest_loading_indices = components.abs().argmax(dim=0)
    signs = torch.sign(components[largest_loading_indices, torch.arange(PCA_COMPONENTS, device=tokens.device)])
    components = components * torch.where(signs < 0, -torch.ones_like(signs), torch.ones_like(signs)).unsqueeze(0)

    projected = centered @ components
    lower_bounds = torch.quantile(projected, PCA_LOWER_QUANTILE, dim=0)
    upper_bounds = torch.quantile(projected, PCA_UPPER_QUANTILE, dim=0)
    total_variance = eigenvalues.clamp_min(0).sum().clamp_min(torch.finfo(torch.float32).eps)
    explained_variance_ratio = selected_values / total_variance
    return PCARGBBasis(
        feature_mean=feature_mean,
        components=components,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        explained_variance_ratio=explained_variance_ratio,
    )


def transform_centered_pca_rgb(tokens: Tensor, tokenized_size: Sequence[int], basis: PCARGBBasis) -> Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != prod(int(size) for size in tokenized_size):
        raise ValueError("visual tokens must match tokenized_size")
    centered = center_visual_tokens(tokens).float() - basis.feature_mean
    projected = centered @ basis.components
    scale = (basis.upper_bounds - basis.lower_bounds).clamp_min(torch.finfo(torch.float32).eps)
    colors = ((projected - basis.lower_bounds) / scale).clamp(0, 1)
    height, width = (int(size) for size in tokenized_size)
    return colors.reshape(tokens.shape[0], height, width, PCA_COMPONENTS).mul(255).round().to(torch.uint8)
