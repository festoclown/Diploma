"""Encoding explicitly downloaded snapshots; never resolve models online."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .data import prepared_dir, verify_prepared
from .downloads import model_dir, verify_model
from .storage import digest, environment, file_hash, new_directory, read_json, read_jsonl, write_json, write_npz


def embedding_dir(root: Path, c: dict) -> Path:
    return prepared_dir(root, c) / ("embedding-" + digest(c["model"])[:16])


def encode(root: Path, c: dict) -> Path:
    if c["dataset"]["kind"] != "text":
        return prepared_dir(root, c)
    prepared = prepared_dir(root, c)
    data_manifest = verify_prepared(prepared)
    spec = c["model"]
    local = model_dir(root, spec)
    model_manifest = verify_model(local)
    signature = digest({"data": data_manifest, "model": model_manifest, "settings": spec})
    path = embedding_dir(root, c)
    if path.exists():
        old = read_json(path / "manifest.json")
        if old["signature"] != signature or file_hash(path / "vectors.npz") != old["sha256"]:
            raise ValueError("Embedding cache mismatch")
        return path
    # Encoding must use the explicit snapshot downloaded in a separate stage;
    # it must not silently fetch a changed model during an experiment.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(local / "snapshot"), device=spec["device"],
                                local_files_only=True, trust_remote_code=False)
    model.max_seq_length = spec["max_length"]
    model.eval()
    texts = [row["text"] for row in read_jsonl(prepared / "documents.jsonl")]
    truncated = 0
    for offset in range(0, len(texts), 128):
        tokens = model.tokenizer(texts[offset:offset + 128], truncation=False,
                                 padding=False, add_special_tokens=True)
        truncated += sum(len(t) > spec["max_length"] for t in tokens["input_ids"])
    x = model.encode(texts, batch_size=spec["batch_size"], normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True)
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) < 1e-10):
        raise ValueError("Invalid embedding vectors")
    new_directory(path)
    write_npz(path / "vectors.npz", x=x)
    write_json(path / "manifest.json", {"signature": signature, "model": model_manifest,
               "settings": spec, "sha256": file_hash(path / "vectors.npz"),
               "dimension": x.shape[1], "truncated_documents": truncated,
               "truncated_fraction": truncated / len(texts), "environment": environment()})
    return path


def vector_path(root: Path, c: dict) -> Path:
    prepared = prepared_dir(root, c)
    data_manifest = verify_prepared(prepared)
    if c["dataset"]["kind"] in {"synthetic", "tabular"}:
        return prepared / "vectors.npz"
    path = embedding_dir(root, c)
    manifest = read_json(path / "manifest.json")
    expected = digest({"data": data_manifest, "model": read_json(model_dir(root, c["model"]) / "manifest.json"),
                       "settings": c["model"]})
    if manifest["signature"] != expected:
        raise ValueError("Embedding provenance differs from prepared data/model")
    if file_hash(path / "vectors.npz") != manifest["sha256"]:
        raise ValueError("Embedding cache corrupted")
    return path / "vectors.npz"
