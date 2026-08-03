import pytest
import torch

from mjepa_cifar10.collapse import EmbeddingCollapseMetric, PatchTokenDiversityMetric


EMBEDDING_DIM = 4
NUM_PATCHES = 4


def test_embedding_collapse_metric_reports_full_rank_symmetric_features() -> None:
    metric = EmbeddingCollapseMetric(EMBEDDING_DIM)
    basis = torch.eye(EMBEDDING_DIM)

    metric.update(torch.cat((basis, -basis)))
    result = metric.compute()

    assert result["std_mean"].item() == pytest.approx(0.5)
    assert result["std_min"].item() == pytest.approx(0.5)
    assert result["effective_rank_fraction"].item() == pytest.approx(1.0)
    assert result["top_eigenvalue_fraction"].item() == pytest.approx(0.25)
    assert result["mean_pairwise_cosine"].item() == pytest.approx(-1 / 7)
    assert result["finite_fraction"].item() == pytest.approx(1.0)


def test_embedding_collapse_metric_detects_constant_features() -> None:
    metric = EmbeddingCollapseMetric(EMBEDDING_DIM)

    metric.update(torch.ones(8, EMBEDDING_DIM))
    result = metric.compute()

    assert result["std_mean"].item() == pytest.approx(0.0)
    assert result["effective_rank_fraction"].item() == pytest.approx(0.0)
    assert result["top_eigenvalue_fraction"].item() == pytest.approx(1.0)
    assert result["mean_pairwise_cosine"].item() == pytest.approx(1.0)


def test_embedding_collapse_metric_tracks_nonfinite_rows() -> None:
    metric = EmbeddingCollapseMetric(EMBEDDING_DIM)
    embeddings = torch.eye(EMBEDDING_DIM)
    embeddings[0, 0] = torch.nan

    metric.update(embeddings)
    result = metric.compute()

    assert result["finite_fraction"].item() == pytest.approx(0.75)


def test_patch_token_diversity_metric_detects_identical_tokens_within_each_image() -> None:
    metric = PatchTokenDiversityMetric(EMBEDDING_DIM)
    image_embeddings = torch.eye(EMBEDDING_DIM).repeat_interleave(NUM_PATCHES, dim=0)
    patches = image_embeddings.reshape(EMBEDDING_DIM, NUM_PATCHES, EMBEDDING_DIM)

    metric.update(patches)
    result = metric.compute()

    assert result["mean_within_image_pairwise_cosine"].item() == pytest.approx(1.0)
    assert result["centered_patch_energy_ratio"].item() == pytest.approx(0.0)
    assert result["centered_patch_effective_rank_fraction"].item() == pytest.approx(0.0)
    assert result["finite_image_fraction"].item() == pytest.approx(1.0)


def test_patch_token_diversity_metric_reports_spatially_diverse_tokens() -> None:
    metric = PatchTokenDiversityMetric(EMBEDDING_DIM)
    basis = torch.eye(EMBEDDING_DIM)
    patches = torch.stack((torch.cat((basis, -basis)), torch.cat((-basis, basis))))

    metric.update(patches)
    result = metric.compute()

    assert result["mean_within_image_pairwise_cosine"].item() == pytest.approx(-1 / 7)
    assert result["centered_patch_energy_ratio"].item() == pytest.approx(1.0)
    assert result["centered_patch_effective_rank_fraction"].item() == pytest.approx(1.0)
    assert result["finite_image_fraction"].item() == pytest.approx(1.0)


def test_patch_token_diversity_metric_tracks_nonfinite_images() -> None:
    metric = PatchTokenDiversityMetric(EMBEDDING_DIM)
    patches = torch.randn(2, NUM_PATCHES, EMBEDDING_DIM)
    patches[0, 0, 0] = torch.nan

    metric.update(patches)
    result = metric.compute()

    assert result["finite_image_fraction"].item() == pytest.approx(0.5)
