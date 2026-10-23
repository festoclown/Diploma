"""NIP parsing and project-level inheritance; no experiment objects in configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from nip import load

from .storage import digest


def merge(base: dict, patch: dict) -> dict:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_configs(path: Path, ancestors: tuple[Path, ...] = ()) -> list[dict]:
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"Config inheritance cycle: {path}")
    parsed = load(str(path))
    # NIP's @ expansion yields several complete variants from one experiment file.
    variants = [parsed] if isinstance(parsed, dict) else list(parsed)
    result = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("NIP must construct dictionaries")
        variant = deepcopy(variant)
        parent = variant.pop("extends", None)
        if parent:
            # Apply the parent separately to each variant so a child override cannot
            # accidentally mutate another condition in a factorial experiment.
            bases = load_configs(path.parent / parent, (*ancestors, path))
            if len(bases) != 1:
                raise ValueError("extends must reference one non-iterable base config")
            variant = merge(bases[0], variant)
        # JSON digest also rejects objects constructed by arbitrary NIP tags.
        digest(variant)
        result.append(variant)
    return result


def validate(c: dict) -> None:
    required = {"name", "phase", "dataset", "model", "compute", "candidates", "selection", "evaluation"}
    if set(c) != required:
        raise ValueError(f"Unexpected/missing top-level keys: {set(c) ^ required}")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", c["name"]):
        raise ValueError("name must be a safe directory name")
    if c["phase"] not in {"pilot", "development", "confirmation", "sensitivity"}:
        raise ValueError("Unknown phase")
    ds, s, p, e = c["dataset"], c["selection"], c["candidates"], c["evaluation"]
    block_keys = {
        "model": {"repo", "revision", "max_length", "batch_size", "device"},
        "compute": {"backend"},
        "candidates": {"k", "algorithms", "n_init", "max_iter", "seed",
                       "hdbscan_min_cluster_size", "hdbscan_min_samples", "spectral_k",
                       "spectral_neighbors", "spectral_n_init", "min_coverage",
                       "projection_dim"},
        "selection": {"methods", "budgets", "scenarios", "repeats", "repeat_seed_start",
                      "bootstrap_draws", "internal", "fixed_window", "adaptive_multiplier",
                      "bias_weight", "fallback"},
        "evaluation": {"split_seed", "test_fraction", "near_optimal_delta", "large_loss_threshold"},
    }
    for block, keys in block_keys.items():
        if set(c[block]) != keys:
            raise ValueError(f"Unexpected/missing {block} keys: {set(c[block]) ^ keys}")
    if c["compute"]["backend"] not in {"cpu", "cuda"}:
        raise ValueError("compute.backend must be cpu or cuda")
    if ds["kind"] not in {"synthetic", "text", "tabular"}:
        raise ValueError("Unknown dataset kind")
    if ds["kind"] == "synthetic":
        expected_dataset = {"kind", "geometry", "balance", "n", "d", "seed"}
    elif ds["kind"] == "text":
        expected_dataset = {"kind", "name", "max_documents", "subsample_seed", "source_revision"}
    else:
        expected_dataset = {"kind", "name", "max_samples", "subsample_seed"}
    if set(ds) != expected_dataset:
        raise ValueError(f"Unexpected/missing dataset keys: {set(ds) ^ expected_dataset}")
    if ds["kind"] == "synthetic":
        if ds["geometry"] not in {"separated", "overlapping", "elongated", "moons"}:
            raise ValueError("Unknown geometry")
        if ds["balance"] not in {"balanced", "imbalanced"} or ds["d"] < 8 or ds["n"] < 16:
            raise ValueError("Invalid synthetic setting")
        if c["phase"] == "development" and ds["seed"] not in [0, 1, 2]:
            raise ValueError("Development is restricted to seeds 0,1,2")
        if c["phase"] in {"confirmation", "sensitivity"} and ds["seed"] < 100:
            raise ValueError("Confirmation cannot reuse development/pilot seeds")
    else:
        allowed = ({"ag_news", "20newsgroups", "banking77"} if ds["kind"] == "text"
                   else {"digits", "breast_cancer", "dry_bean"})
        if ds["name"] not in allowed:
            raise ValueError("Unknown real dataset")
        maximum = ds["max_documents"] if ds["kind"] == "text" else ds["max_samples"]
        if type(maximum) is not int or maximum < 16:
            raise ValueError("Invalid real dataset size")
        if c["phase"] == "development":
            raise ValueError("Real datasets may not tune M4")
    if s["internal"] not in {"silhouette", "davies_bouldin"}:
        raise ValueError("Unknown internal score")
    if not s["budgets"] or any(type(x) is not int or x < 2 for x in s["budgets"]):
        raise ValueError("Budgets must be integers >=2")
    if s["repeats"] < 1 or s["bootstrap_draws"] < 2:
        raise ValueError("Invalid repeat/bootstrap counts")
    if not set(s["methods"]) <= {f"M{i}" for i in range(8)} or not s["methods"]:
        raise ValueError("Unknown or empty methods")
    if not s["scenarios"] or not set(s["scenarios"]) <= {"random", "biased", "missing_classes"}:
        raise ValueError("Unknown annotation scenario")
    if s["fixed_window"] < 0 or s["adaptive_multiplier"] < 0 or s["bias_weight"] <= 0:
        raise ValueError("Invalid selection settings")
    if not p["k"] or any(type(k) is not int or k < 2 for k in p["k"]):
        raise ValueError("K must be integer >=2")
    if not p["algorithms"] or len(set(p["algorithms"])) != len(p["algorithms"]):
        raise ValueError("Candidate algorithms must be a nonempty unique list")
    if not set(p["algorithms"]) <= {"kmeans", "hdbscan", "spectral"}:
        raise ValueError("Unknown clustering algorithm")
    if (not p["spectral_k"] or not set(p["spectral_k"]) <= set(p["k"]) or
            any(type(k) is not int for k in p["spectral_k"])):
        raise ValueError("spectral_k must be a nonempty subset of k")
    if (not p["hdbscan_min_cluster_size"] or
            any(type(k) is not int or k < 2 for k in p["hdbscan_min_cluster_size"])):
        raise ValueError("Invalid HDBSCAN sizes")
    if (p["hdbscan_min_samples"] < 3 or p["spectral_neighbors"] < 2 or
            p["spectral_n_init"] < 1 or p["n_init"] < 1 or p["max_iter"] < 1 or
            not 0 < p["min_coverage"] <= 1 or p["projection_dim"] < 0):
        raise ValueError("Invalid candidate settings")
    for key in ["methods", "budgets", "scenarios"]:
        if len(set(s[key])) != len(s[key]):
            raise ValueError(f"Duplicate {key} entries")
    if type(s["fallback"]) is not bool:
        raise ValueError("fallback must be boolean")
    if not 0 < e["test_fraction"] < 1:
        raise ValueError("Invalid test_fraction")


def dataset_key(c: dict) -> str:
    d = c["dataset"]
    label = d["name"] if d["kind"] != "synthetic" else "synthetic"
    return f"{label}-{digest(d)[:16]}"
