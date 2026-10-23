"""Explicit NumPy/CuPy boundary for the expensive vector operations."""

from __future__ import annotations

import numpy as np


class Backend:
    def __init__(self, name: str):
        self.name = name
        if name == "cpu":
            self.xp = np
        elif name == "cuda":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("No CUDA device is visible")
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError("CUDA backend requires a compatible CuPy/RAPIDS environment") from exc
            self.xp = cp
        else:
            raise ValueError(f"Unknown backend: {name}")

    def array(self, value):
        return self.xp.asarray(value)

    def host(self, value) -> np.ndarray:
        return np.asarray(value) if self.name == "cpu" else self.xp.asnumpy(value)

    def scalar(self, value) -> float:
        return float(value) if self.name == "cpu" else float(value.item())

    def project(self, x, dimensions: int):
        if not dimensions or dimensions >= x.shape[1]:
            return x
        if self.name == "cuda":
            from cuml.decomposition import PCA
            projected = PCA(n_components=dimensions, output_type="cupy").fit_transform(x)
        else:
            from sklearn.decomposition import PCA
            projected = PCA(n_components=dimensions, svd_solver="full").fit_transform(x)
        return self.xp.asarray(projected, dtype=self.xp.float32)

    def distances(self, x, metric: str):
        xp = self.xp
        x = xp.asarray(x, dtype=xp.float32)
        # Compute one dense distance matrix on the selected device and reuse it
        # when scoring every candidate partition of this same dataset.
        if metric == "cosine":
            norms = xp.linalg.norm(x, axis=1, keepdims=True)
            unit = x / xp.maximum(norms, 1e-12)
            result = xp.maximum(1 - unit @ unit.T, 0)
        elif metric == "euclidean":
            squared = xp.sum(x * x, axis=1)
            result = xp.sqrt(xp.maximum(squared[:, None] + squared[None, :]
                                        - 2 * (x @ x.T), 0))
        else:
            raise ValueError(metric)
        result = (result + result.T) / 2
        xp.fill_diagonal(result, 0)
        return result

    def internal(self, x, distance, labels: np.ndarray) -> tuple[float, float, float]:
        """Full-pool scores. Noise is excluded then penalized by assigned coverage."""
        xp = self.xp
        # HDBSCAN's -1 is unassigned noise. It must not become an ordinary
        # cluster when the geometric indices are computed.
        assigned = np.asarray(labels) != -1
        coverage = float(assigned.mean())
        codes = np.unique(np.asarray(labels)[assigned], return_inverse=True)[1]
        if len(codes) < 3 or not 2 <= len(np.unique(codes)) < len(codes):
            raise ValueError("Silhouette/DB need at least two nonsingleton groups")
        indices = xp.asarray(np.flatnonzero(assigned))
        z = x[indices]
        d = distance[xp.ix_(indices, indices)]
        codes = xp.asarray(codes, dtype=xp.int32)
        n, k = len(codes), int(self.host(codes.max())) + 1
        membership = xp.eye(k, dtype=xp.float32)[codes]
        counts = membership.sum(axis=0)
        sums = d @ membership
        mean_distance = sums / xp.maximum(counts[None, :], 1)
        row = xp.arange(n)
        own = sums[row, codes] / xp.maximum(counts[codes] - 1, 1)
        mean_distance[row, codes] = xp.inf
        other = mean_distance.min(axis=1)
        denominator = xp.maximum(own, other)
        silhouettes = xp.where((counts[codes] > 1) & (denominator > 0),
                               (other - own) / xp.maximum(denominator, 1e-12), 0)
        # Penalize candidates that obtain a high silhouette by rejecting many
        # difficult points; KMeans and Spectral have coverage == 1.
        silhouette = coverage * self.scalar(silhouettes.mean())

        centroids = (membership.T @ z) / counts[:, None]
        scatter_per_point = xp.linalg.norm(z - centroids[codes], axis=1)
        scatter = (membership.T @ scatter_per_point) / counts
        centroid_distance = self.distances(centroids, "euclidean")
        ratio = (scatter[:, None] + scatter[None, :]) / xp.maximum(
            centroid_distance, 1e-12)
        xp.fill_diagonal(ratio, -xp.inf)
        db = self.scalar(ratio.max(axis=1).mean()) / coverage
        return silhouette, db, coverage
