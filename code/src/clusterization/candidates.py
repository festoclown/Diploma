"""GPU-resident candidate construction; labels only return to host for artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backend import Backend
from .config import dataset_key
from .embeddings import vector_path
from .storage import digest, environment, file_hash, new_directory, read_json, write_json, write_npz


def canonical(labels: np.ndarray) -> np.ndarray:
    """Keep HDBSCAN's -1 noise label distinct from genuine cluster IDs."""
    mapping = {}
    result = []
    for value in labels:
        value = int(value)
        result.append(-1 if value == -1 else mapping.setdefault(value, len(mapping)))
    return np.asarray(result, dtype=np.int32)


def pool_location(root: Path, c: dict) -> tuple[Path, dict]:
    vectors = vector_path(root, c)
    # The pool identity depends on vectors, algorithm settings and backend,
    # not on the later rule used to select a candidate. Sensitivity runs can
    # therefore compare selectors on exactly the same cached partitions.
    spec = {"schema": 2, "vectors_sha256": file_hash(vectors),
            "candidates": c["candidates"], "compute": c["compute"],
            "distance": "cosine" if c["dataset"]["kind"] == "text" else "euclidean",
            "noise_policy": "assigned-only internal scores penalized by coverage; noise singletons in ARI"}
    return root / "artifacts" / "pools" / dataset_key(c) / digest(spec)[:16], spec


def _partition(backend: Backend, x, algorithm: str, parameter: int, params: dict):
    if backend.name == "cuda":
        from cuml.cluster import HDBSCAN, KMeans, SpectralClustering

        common = {"output_type": "cupy"}
        if algorithm == "kmeans":
            model = KMeans(n_clusters=parameter, n_init=params["n_init"],
                           max_iter=params["max_iter"], random_state=params["seed"],
                           init="k-means++", **common)
        elif algorithm == "hdbscan":
            model = HDBSCAN(min_cluster_size=parameter,
                            # cuML excludes the point itself; sklearn includes it.
                            min_samples=params["hdbscan_min_samples"] - 1,
                            metric="euclidean", build_algo="brute_force", **common)
        else:
            model = SpectralClustering(n_clusters=parameter,
                n_neighbors=min(params["spectral_neighbors"], len(x) - 1),
                n_init=params["spectral_n_init"], random_state=params["seed"],
                affinity="nearest_neighbors", **common)
        return backend.host(model.fit_predict(x))

    from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering

    if algorithm == "kmeans":
        model = KMeans(n_clusters=parameter, n_init=params["n_init"],
                       max_iter=params["max_iter"], random_state=params["seed"])
    elif algorithm == "hdbscan":
        model = HDBSCAN(min_cluster_size=parameter,
                        min_samples=params["hdbscan_min_samples"], metric="euclidean")
    else:
        model = SpectralClustering(n_clusters=parameter,
            n_neighbors=min(params["spectral_neighbors"], len(x) - 1),
            n_init=params["spectral_n_init"], random_state=params["seed"],
            affinity="nearest_neighbors", assign_labels="kmeans")
    return np.asarray(model.fit_predict(x))


def build_pool(root: Path, c: dict) -> Path:
    path, spec = pool_location(root, c)
    if path.exists():
        load_pool(root, c)
        return path
    with np.load(vector_path(root, c), allow_pickle=False) as data:
        features = np.asarray(data["x"], dtype=np.float32)
    if not np.isfinite(features).all() or len(features) <= max(c["candidates"]["k"]):
        raise ValueError("Need finite features and K < n")
    backend = Backend(c["compute"]["backend"])
    x = backend.array(features)
    params = c["candidates"]
    clustering_x = backend.project(x, params["projection_dim"])
    if c["dataset"]["kind"] == "text" and clustering_x is not x:
        xp = backend.xp
        clustering_x = clustering_x / xp.maximum(
            xp.linalg.norm(clustering_x, axis=1, keepdims=True), 1e-12)
    distance = backend.distances(x, spec["distance"])
    unique, skipped = {}, []
    grids = {"kmeans": params["k"], "hdbscan": params["hdbscan_min_cluster_size"],
             "spectral": params["spectral_k"]}
    for algorithm in params["algorithms"]:
        for parameter in grids[algorithm]:
            alias = ({"algorithm": algorithm, "requested_k": parameter} if algorithm != "hdbscan"
                     else {"algorithm": algorithm, "min_cluster_size": parameter,
                           "min_samples": params["hdbscan_min_samples"]})
            # Canonical labels make identical partitions recognizable even if
            # two algorithms assign different integer names to their clusters.
            partition = canonical(_partition(backend, clustering_x, algorithm, parameter, params))
            coverage = float(np.mean(partition != -1))
            cluster_count = int(len(np.unique(partition[partition != -1])))
            if coverage < params["min_coverage"] or not 2 <= cluster_count < int(
                    np.sum(partition != -1)):
                skipped.append({**alias, "reason": "coverage or cluster-count requirement",
                                "coverage": coverage, "clusters": cluster_count})
                continue
            identifier = digest(partition.tolist())
            if identifier in unique:
                unique[identifier]["aliases"].append(alias)
                continue
            silhouette, db, _ = backend.internal(x, distance, partition)
            if not np.isfinite([silhouette, db]).all():
                skipped.append({**alias, "reason": "nonfinite internal score"})
                continue
            unique[identifier] = {"id": identifier, "aliases": [alias],
                "partition": partition, "k": cluster_count, "coverage": coverage,
                "silhouette": silhouette, "davies_bouldin": db}
    if not unique:
        raise ValueError("No valid candidate partitions; inspect skipped settings")
    ordered = [unique[key] for key in sorted(unique)]
    labels = np.stack([row.pop("partition") for row in ordered])
    new_directory(path)
    write_npz(path / "pool.npz", labels=labels)
    write_json(path / "manifest.json", {"spec": spec, "candidates": ordered,
               "skipped": skipped, "projection_used": (params["projection_dim"]
                 if 0 < params["projection_dim"] < features.shape[1] else 0),
               "sha256": file_hash(path / "pool.npz"), "environment": environment()})
    return path


def load_pool(root: Path, c: dict) -> tuple[np.ndarray, dict]:
    path, spec = pool_location(root, c)
    manifest = read_json(path / "manifest.json")
    if manifest["spec"] != spec or file_hash(path / "pool.npz") != manifest["sha256"]:
        raise ValueError("Pool cache mismatch")
    with np.load(path / "pool.npz", allow_pickle=False) as data:
        labels = data["labels"]
    return labels, manifest


def internal_scores(manifest: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([row[name] for row in manifest["candidates"]])
    return (values, (values + 1) / 2) if name == "silhouette" else (-values, 1 / (1 + values))
