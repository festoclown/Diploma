"""Scientific correctness checks; supplied for manual execution, not run at creation."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, silhouette_score

from clusterization.backend import Backend
from clusterization.candidates import canonical
from clusterization.metrics import ari_grid, noise_as_singletons
from clusterization.sampling import bootstrap_weights


@pytest.mark.parametrize("seed", [7, 31, 91])
def test_multiplicity_ari_matches_explicit_object_resampling(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, 25)
    partitions = rng.integers(0, 6, (5, 25))
    groups = np.array([i // 3 for i in range(25)])
    weights = bootstrap_weights(groups, 12, rng)
    actual = ari_grid(y, partitions, weights)
    for b, w in enumerate(weights):
        indices = np.repeat(np.arange(len(y)), w)
        expected = [adjusted_rand_score(y[indices], p[indices]) for p in partitions]
        np.testing.assert_allclose(actual[b], expected, atol=1e-12)


@pytest.mark.parametrize("y", [[0, 0, 0, 0], [0, 1, 2, 3], [0, 0, 1, 1]])
def test_ari_degenerate_and_perfect_cases_match_sklearn(y):
    y = np.asarray(y)
    partitions = np.array([[0, 0, 0, 0], [0, 1, 2, 3], y])
    expected = [adjusted_rand_score(y, p) for p in partitions]
    np.testing.assert_allclose(ari_grid(y, partitions)[0], expected, atol=1e-12)


def test_hdbscan_noise_is_not_a_single_cluster_for_external_scores():
    y = np.array([0, 0, 1, 1, 2, 2])
    labels = np.array([4, 4, 7, 7, -1, -1])
    assert canonical(labels).tolist() == [0, 0, 1, 1, -1, -1]
    expected = adjusted_rand_score(y, noise_as_singletons(labels))
    np.testing.assert_allclose(ari_grid(y, labels[None, :])[0, 0], expected)


def test_vector_internal_scores_match_sklearn_without_noise():
    x = np.array([[0, 0], [0.2, 0], [0, 0.2],
                  [3, 3], [3.2, 3], [3, 3.2]], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1])
    backend = Backend("cpu")
    silhouette, db, coverage = backend.internal(
        x, backend.distances(x, "euclidean"), labels)
    np.testing.assert_allclose(silhouette, silhouette_score(x, labels), atol=1e-5)
    np.testing.assert_allclose(db, davies_bouldin_score(x, labels), atol=1e-5)
    assert coverage == 1.0
