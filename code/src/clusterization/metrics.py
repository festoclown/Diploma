"""External scores and ARI on resampled objects (not independent pairs)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score, completeness_score, homogeneity_score,
)

from .backend import Backend


def noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).copy()
    noise = labels == -1
    if np.any(noise):
        labels[noise] = np.arange(int(noise.sum())) + int(labels.max()) + 1
    return labels


def _pair_counts(codes, weights, xp):
    """All bootstrap draws and candidate contingency tables in one bincount."""
    candidates, objects = codes.shape
    draws = weights.shape[0]
    cells = int(codes.max()) + 1
    # Offsets give each (bootstrap draw, candidate) its own bincount segment.
    # This avoids a Python loop over candidates while retaining exact integer
    # multiplicities of resampled objects.
    offsets = cells * xp.arange(draws * candidates).reshape(draws, candidates, 1)
    flat_codes = (codes[None, :, :] + offsets).ravel()
    flat_weights = xp.broadcast_to(weights[:, None, :],
                                   (draws, candidates, objects)).ravel()
    counts = xp.bincount(flat_codes, weights=flat_weights,
                         minlength=draws * candidates * cells)
    counts = counts.reshape(draws, candidates, cells)
    return (counts * (counts - 1) / 2).sum(axis=2)


def ari_grid(y: np.ndarray, partitions: np.ndarray, weights: np.ndarray | None = None,
             backend: Backend | None = None
             ) -> np.ndarray:
    """Return [resamples, candidates] ARI, matching explicit object repetition.

    partitions is [candidates, objects]; integer weights count object repetitions.
    This is NOT an importance-weighted ARI estimator for sampling bias.
    """
    y = np.asarray(y)
    partitions = np.asarray(partitions)
    if y.ndim != 1 or partitions.ndim != 2 or partitions.shape[1] != len(y) or not len(y):
        raise ValueError("Expected nonempty y[n] and partitions[candidates,n]")
    if weights is None:
        weights = np.ones((1, len(y)), dtype=np.int64)
    weights = np.asarray(weights)
    if (weights.ndim != 2 or weights.shape[1] != len(y) or
            not np.all(np.isfinite(weights)) or np.any(weights < 0) or
            np.any(weights != np.floor(weights))):
        raise ValueError("Bootstrap weights must be nonnegative integer multiplicities")
    backend = backend or Backend("cpu")
    xp = backend.xp
    yi = np.unique(y, return_inverse=True)[1].astype(np.int64)
    coded = []
    for row in partitions:
        coded.append(np.unique(noise_as_singletons(row), return_inverse=True)[1])
    # Each noise point is an unassigned singleton, not a common "-1 cluster".
    pi = np.stack(coded).astype(np.int64)
    categories = int(pi.max()) + 1
    joint = yi[None, :] * categories + pi
    device_weights = xp.asarray(weights, dtype=xp.float64)
    a = _pair_counts(xp.asarray(yi[None, :]), device_weights, xp)
    b = _pair_counts(xp.asarray(pi), device_weights, xp)
    both = _pair_counts(xp.asarray(joint), device_weights, xp)
    n = device_weights.sum(axis=1)
    pairs = n * (n - 1) / 2
    # CuPy's divide does not accept NumPy's where= keyword. Safe denominators
    # avoid invalid divisions in both backends before xp.where selects values.
    has_pairs = pairs[:, None] > 0
    safe_pairs = xp.where(has_pairs, pairs[:, None], 1.0)
    expected = xp.where(has_pairs, (a * b) / safe_pairs, 0.0)
    denominator = (a + b) / 2 - expected
    has_denominator = xp.abs(denominator) > 1e-12
    safe_denominator = xp.where(has_denominator, denominator, 1.0)
    result = xp.where(has_denominator,
                      (both - expected) / safe_denominator, 1.0)
    return backend.host(result)


def external_scores(y: np.ndarray, partitions: np.ndarray,
                    backend: Backend | None = None) -> list[dict]:
    ari = ari_grid(y, partitions, backend=backend)[0]
    return [{"ari": float(ari[i]),
             "ami": float(adjusted_mutual_info_score(
                 y, noise_as_singletons(p), average_method="arithmetic")),
             "homogeneity": float(homogeneity_score(y, noise_as_singletons(p))),
             "completeness": float(completeness_score(y, noise_as_singletons(p)))}
            for i, p in enumerate(partitions)]


def positive_pairs(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))
