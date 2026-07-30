import pytest
import torch

from mjepa_cifar10.collapse import EmbeddingCollapseMetric


EMBEDDING_DIM = 4


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
