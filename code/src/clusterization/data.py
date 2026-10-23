"""Data preparation, grouping and synthetic generation. No network calls."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, make_moons
from sklearn.preprocessing import StandardScaler

from .config import dataset_key
from .storage import (digest, environment, file_hash, new_directory, read_json,
                      read_jsonl, write_json, write_jsonl, write_npz)


def prepared_dir(root: Path, c: dict) -> Path:
    return root / "data" / "processed" / dataset_key(c)


def synthetic(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec["seed"])
    n, d = spec["n"], spec["d"]
    probs = np.ones(8) if spec["balance"] == "balanced" else 2.0 ** (-np.arange(8) / 2)
    ideal = n * probs / probs.sum()
    sizes = np.floor(ideal).astype(int)
    sizes[np.argsort(-(ideal - sizes), kind="stable")[:n - sizes.sum()]] += 1
    if spec["geometry"] == "moons":
        vectors, labels = [], []
        for pair in range(4):
            k = 2 * pair
            plane, local = make_moons(n_samples=(int(sizes[k]), int(sizes[k + 1])),
                                      noise=0.08, random_state=spec["seed"] + pair)
            plane += np.array([6 * (pair % 2), 6 * (pair // 2)])
            extra = rng.normal(scale=0.04, size=(len(plane), d - 2))
            vectors.append(np.column_stack([plane, extra]))
            labels.append(local + k)
        order = rng.permutation(n)
        return (np.concatenate(vectors)[order].astype(np.float32),
                np.concatenate(labels)[order].astype(np.int32))
    vectors, labels = [], []
    for k, size in enumerate(sizes):
        mean = np.zeros(d)
        mean[k] = 3 if spec["geometry"] == "overlapping" else 6
        x = rng.normal(size=(size, d))
        if spec["geometry"] == "elongated":
            q, _ = np.linalg.qr(rng.normal(size=(d, d)))
            x[:, 0] *= 3
            x = x @ q.T
        vectors.append(x + mean)
        labels.extend([k] * size)
    order = rng.permutation(n)
    return np.concatenate(vectors)[order].astype(np.float32), np.asarray(labels)[order]


def tabular(root: Path, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Public numeric benchmarks; labels are never passed to feature processing."""
    name = spec["name"]
    if name == "digits":
        x, labels = load_digits(return_X_y=True)
    elif name == "breast_cancer":
        x, labels = load_breast_cancer(return_X_y=True)
    else:
        raw = root / "data" / "raw" / "dry_bean"
        source = read_json(raw / "manifest.json")
        if file_hash(raw / "records.npz") != source["records_sha256"]:
            raise ValueError("Dry Bean raw data hash mismatch")
        with np.load(raw / "records.npz", allow_pickle=False) as data:
            x, labels = data["x"], data["y"]
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if not np.isfinite(x).all():
        raise ValueError("Tabular data contain nonfinite features")
    # Identical rows with conflicting labels cannot be used as a clean reference.
    unique = {}
    conflicts = set()
    for i, row in enumerate(x):
        key = row.tobytes()
        if key in unique and labels[unique[key]] != labels[i]:
            conflicts.add(key)
        else:
            unique.setdefault(key, i)
    keep = np.array([i for key, i in unique.items() if key not in conflicts], dtype=int)
    keep.sort()
    if len(keep) > spec["max_samples"]:
        keep = np.sort(np.random.default_rng(spec["subsample_seed"]).choice(
            keep, spec["max_samples"], replace=False))
    if len(keep) < 16:
        raise ValueError("Too few distinct tabular rows")
    # The study is transductive: scaling is fitted without labels on the full X.
    features = StandardScaler().fit_transform(x[keep]).astype(np.float32)
    y = np.unique(labels[keep], return_inverse=True)[1].astype(np.int32)
    ids = np.array([f"{name}:{i}" for i in keep])
    return features, y, ids, {"n": len(y), "d": features.shape[1],
                              "classes": int(len(np.unique(y))),
                              "duplicates_removed": int(len(x) - len(unique)),
                              "conflicting_duplicate_patterns": len(conflicts)}


def clean_records(records: list[dict], maximum: int, seed: int) -> tuple[list[dict], dict]:
    by_text = defaultdict(list)
    empty = 0
    for row in records:
        text = re.sub(r"\s+", " ", row["text"]).strip()
        if not text:
            empty += 1
            continue
        by_text[text].append({**row, "text": text})
    # Union groups of exact duplicates before selecting one representative.
    # This preserves known discussion links even when a duplicate is discarded.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rows in by_text.values():
        groups = [str(r.get("group", r["id"])) for r in rows]
        for group in groups[1:]:
            a, b = find(groups[0]), find(group)
            parent[max(a, b)] = min(a, b)
    clean, conflict, duplicates = [], 0, 0
    for text, rows in by_text.items():
        if len({str(r["label"]) for r in rows}) != 1:
            conflict += len(rows)
            continue
        duplicates += len(rows) - 1
        representative = min(rows, key=lambda r: str(r["id"]))
        clean.append({"id": str(representative["id"]), "text": text,
                      "label": str(representative["label"]),
                      "group": find(str(representative.get("group", representative["id"])))})
    clean.sort(key=lambda r: r["id"])
    available = len(clean)
    if available > maximum:
        indices = np.sort(np.random.default_rng(seed).choice(available, maximum, replace=False))
        clean = [clean[i] for i in indices]
    if len(clean) < 4:
        raise ValueError("Too few documents after cleaning")
    return clean, {"raw_count": len(records), "empty": empty,
                   "conflicting_duplicate_rows": conflict, "duplicates_removed": duplicates,
                   "clean_before_subsample": available, "selected": len(clean)}


def verify_prepared(path: Path) -> dict:
    manifest = read_json(path / "manifest.json")
    for name, expected in manifest["files"].items():
        if file_hash(path / name) != expected:
            raise ValueError(f"Prepared file changed: {path / name}")
    return manifest


def prepare(root: Path, c: dict) -> Path:
    path = prepared_dir(root, c)
    spec = c["dataset"]
    source = None
    if spec["kind"] == "text":
        raw = root / "data" / "raw" / spec["name"]
        source = read_json(raw / "manifest.json")
        if file_hash(raw / "records.jsonl") != source["records_sha256"]:
            raise ValueError("Raw corpus hash mismatch")
    elif spec["kind"] == "tabular" and spec["name"] == "dry_bean":
        raw = root / "data" / "raw" / "dry_bean"
        source = read_json(raw / "manifest.json")
        if file_hash(raw / "records.npz") != source["records_sha256"]:
            raise ValueError("Dry Bean raw data hash mismatch")
    elif spec["kind"] == "tabular":
        from sklearn import __version__ as sklearn_version
        source = {"loader": f"sklearn.datasets.load_{spec['name']}",
                  "sklearn_version": sklearn_version}
    signature = digest({"schema": 1, "spec": spec, "source": source})
    if path.exists():
        previous = verify_prepared(path)
        if previous["signature"] != signature:
            raise ValueError("Prepared cache differs; use a new project copy for new source revisions")
        return path
    new_directory(path)
    if spec["kind"] == "synthetic":
        x, y = synthetic(spec)
        groups = np.arange(len(y)).astype(str)
        write_npz(path / "vectors.npz", x=x)
        ids = np.array([f"synthetic:{i}" for i in range(len(y))])
        metadata = {"n": len(y), "d": x.shape[1]}
        files = ["vectors.npz", "labels.npz"]
    elif spec["kind"] == "tabular":
        x, y, ids, metadata = tabular(root, spec)
        groups = ids.copy()
        write_npz(path / "vectors.npz", x=x)
        files = ["vectors.npz", "labels.npz"]
    else:
        rows, metadata = clean_records(read_jsonl(raw / "records.jsonl"),
                                       spec["max_documents"], spec["subsample_seed"])
        names, y = np.unique([r["label"] for r in rows], return_inverse=True)
        groups = np.array([r["group"] for r in rows])
        ids = np.array([r["id"] for r in rows])
        write_jsonl(path / "documents.jsonl", [{"id": r["id"], "text": r["text"]} for r in rows])
        metadata["class_names"] = names.tolist()
        files = ["documents.jsonl", "labels.npz"]
    write_npz(path / "labels.npz", y=y, groups=groups, ids=ids)
    write_json(path / "manifest.json", {"signature": signature, "dataset": spec,
               "source": source, "statistics": metadata, "environment": environment(),
               "files": {name: file_hash(path / name) for name in files}})
    return path
