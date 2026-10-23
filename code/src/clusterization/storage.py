"""Explicit, atomic artifact IO and deterministic provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np


def digest(value: Any) -> str:
    def numpy_scalar(scalar: Any) -> Any:
        if isinstance(scalar, np.generic):
            return scalar.item()
        raise TypeError(f"Object of type {type(scalar).__name__} is not JSON serializable")

    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, default=numpy_scalar).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def derived_seed(*parts: Any) -> int:
    return int(digest(list(parts))[:8], 16)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def new_directory(path: Path) -> None:
    """Never overwrite a completed or partial run."""
    path.mkdir(parents=True, exist_ok=False)


def environment() -> dict:
    names = ["nip-config", "numpy", "scipy", "scikit-learn", "pandas", "sentence-transformers",
             "torch", "datasets", "huggingface-hub", "cupy-cuda12x", "cuml-cu12",
             "clusterization-research"]
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "source_sha256": source}


def resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Project path escapes root: {relative}")
    return path
