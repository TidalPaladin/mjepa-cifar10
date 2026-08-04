import pytest
import torch

from mjepa_cifar10.patch_spatial_diagnostics import (
    distance_binned_cosine_sums,
    fit_centered_pca_rgb,
    manhattan_distance_matrix,
    transform_centered_pca_rgb,
)


def test_manhattan_distance_matrix_matches_two_by_two_grid() -> None:
    distances = manhattan_distance_matrix((2, 2))

    assert distances.tolist() == [
        [0, 1, 1, 2],
        [1, 0, 2, 1],
        [1, 2, 0, 1],
        [2, 1, 1, 0],
    ]


def test_distance_binned_cosine_sums_use_unique_pairs() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])

    sums, counts = distance_binned_cosine_sums(tokens, (2, 2), center_within_image=False)

    assert counts.tolist() == [4, 2]
    assert sums.tolist() == pytest.approx([2.0, 0.0])


def test_centered_pca_rgb_has_expected_shape_and_range() -> None:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(6, 4, 5, generator=generator)
    basis = fit_centered_pca_rgb(tokens)

    colors = transform_centered_pca_rgb(tokens[:2], (2, 2), basis)

    assert colors.shape == (2, 2, 2, 3)
    assert colors.dtype == torch.uint8
    assert basis.explained_variance_ratio.shape == (3,)
    assert torch.all(basis.explained_variance_ratio >= 0)
