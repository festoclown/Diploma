"""Network access is confined to explicitly invoked download commands."""

from __future__ import annotations

import csv
import hashlib
from email.parser import Parser
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io import arff

from .storage import (digest, environment, file_hash, new_directory, read_json,
                      write_json, write_jsonl, write_npz)


def model_dir(root: Path, spec: dict) -> Path:
    return root / "models" / digest({"repo": spec["repo"], "revision": spec["revision"]})[:16]


def download_model(root: Path, spec: dict) -> Path:
    from huggingface_hub import HfApi, snapshot_download

    path = model_dir(root, spec)
    if path.exists():
        verify_model(path)
        return path
    # Resolve a moving branch such as main to a commit before downloading;
    # the manifest records that exact model revision for reproducibility.
    revision = HfApi().model_info(spec["repo"], revision=spec["revision"]).sha
    new_directory(path)
    snapshot_download(repo_id=spec["repo"], revision=revision, local_dir=path / "snapshot",
                      ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.msgpack"])
    files = {str(p.relative_to(path)): file_hash(p)
             for p in sorted((path / "snapshot").rglob("*"))
             if p.is_file() and ".cache" not in p.parts}
    write_json(path / "manifest.json", {"repo": spec["repo"], "requested_revision": spec["revision"],
               "resolved_revision": revision, "files": files, "environment": environment()})
    return path


def verify_model(path: Path) -> dict:
    manifest = read_json(path / "manifest.json")
    for relative, expected in manifest["files"].items():
        if file_hash(path / relative) != expected:
            raise ValueError(f"Model file changed: {relative}")
    return manifest


def _get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "clusterization-research/0.1"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def _discussion_groups(messages: list[str]) -> list[str]:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, message in enumerate(messages):
        headers = Parser().parsestr(message, headersonly=True)
        ids = re.findall(r"<[^<>\s]+>", " ".join(str(headers.get(k, "")) for k in
                                                 ["Message-ID", "References", "In-Reply-To"]))
        node = f"document:{i}"
        for identifier in ids:
            a, b = find(node), find(identifier)
            parent[max(a, b)] = min(a, b)
    return [find(f"document:{i}") for i in range(len(messages))]


def download_data(root: Path, spec: dict) -> Path | None:
    if spec["kind"] == "tabular" and spec["name"] == "dry_bean":
        path = root / "data" / "raw" / "dry_bean"
        if path.exists():
            manifest = read_json(path / "manifest.json")
            if file_hash(path / "records.npz") != manifest["records_sha256"]:
                raise ValueError("Dry Bean cache mismatch")
            return path
        url = "https://archive.ics.uci.edu/static/public/602/dry%2Bbean%2Bdataset.zip"
        archive_bytes = _get(url)
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            matches = [name for name in archive.namelist()
                       if name.endswith("Dry_Bean_Dataset.arff")]
            if len(matches) != 1:
                raise ValueError("Unexpected Dry Bean archive layout")
            table, _ = arff.loadarff(io.BytesIO(archive.read(matches[0])))
        columns = [name for name in table.dtype.names if name != "Class"]
        if len(columns) != 16:
            raise ValueError("Expected 16 Dry Bean numeric features")
        x = np.column_stack([table[name] for name in columns]).astype(np.float64)
        y = np.array([value.decode("utf-8") for value in table["Class"]])
        if x.shape != (13611, 16) or len(np.unique(y)) != 7:
            raise ValueError("Unexpected Dry Bean data dimensions")
        new_directory(path)
        write_npz(path / "records.npz", x=x, y=y)
        write_json(path / "manifest.json", {"source": url, "uci_dataset_id": 602,
                   "doi": "10.24432/C50S4B",
                   "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                   "records_sha256": file_hash(path / "records.npz"),
                   "count": len(y), "environment": environment()})
        return path
    if spec["kind"] != "text":
        return None
    name = spec["name"]
    path = root / "data" / "raw" / name
    if path.exists():
        manifest = read_json(path / "manifest.json")
        if manifest.get("requested_revision") != spec["source_revision"]:
            raise ValueError("Raw source revision differs; use a new project copy for a new revision")
        if file_hash(path / "records.jsonl") != manifest["records_sha256"]:
            raise ValueError("Raw data hash mismatch")
        return path
    records = []
    if name == "ag_news":
        from datasets import load_dataset
        from huggingface_hub import HfApi

        repo = "fancyzhx/ag_news"
        revision = HfApi().dataset_info(repo, revision=spec["source_revision"]).sha
        data = load_dataset(repo, revision=revision, cache_dir=str(root / "data" / "download_cache"))
        # The original train/test split is not used as our evaluation split:
        # the project creates its own group-aware transductive S/T split.
        for split in ["train", "test"]:
            for i, row in enumerate(data[split]):
                identifier = f"{split}:{i}"
                records.append({"id": identifier, "group": identifier,
                                "text": row["text"], "label": str(row["label"])})
        source = {"repo": repo, "revision": revision, "grouping": "exact_duplicates_only"}
    elif name == "banking77":
        revision = spec["source_revision"]
        if revision == "main":
            revision = "master"
        commit = json.loads(_get("https://api.github.com/repos/PolyAI-LDN/"
                                f"task-specific-datasets/commits/{revision}"))["sha"]
        urls = []
        for split in ["train", "test"]:
            url = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
                   f"{commit}/banking_data/{split}.csv")
            urls.append(url)
            for i, row in enumerate(csv.DictReader(io.StringIO(_get(url).decode("utf-8-sig")))):
                identifier = f"{split}:{i}"
                records.append({"id": identifier, "group": identifier,
                                "text": row["text"], "label": row["category"]})
        source = {"urls": urls, "revision": commit, "grouping": "exact_duplicates_only"}
    elif name == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups

        home = str(root / "data" / "download_cache")
        raw = fetch_20newsgroups(subset="all", data_home=home, shuffle=False)
        clean = fetch_20newsgroups(subset="all", data_home=home, shuffle=False,
                                  remove=("headers", "footers", "quotes"))
        if list(raw.filenames) != list(clean.filenames):
            raise ValueError("Raw and cleaned 20 Newsgroups ordering differs")
        groups = _discussion_groups(raw.data)
        for i, text in enumerate(clean.data):
            records.append({"id": "/".join(Path(raw.filenames[i]).parts[-2:]),
                            "group": groups[i], "text": text,
                            "label": raw.target_names[int(raw.target[i])]})
        source = {"loader": "sklearn.datasets.fetch_20newsgroups", "remove":
                  ["headers", "footers", "quotes"], "grouping": "header_reference_components"}
    else:
        raise ValueError(f"Unknown corpus: {name}")
    new_directory(path)
    write_jsonl(path / "records.jsonl", records)
    write_json(path / "manifest.json", {"name": name, "source": source,
               "requested_revision": spec["source_revision"],
               "records_sha256": file_hash(path / "records.jsonl"), "count": len(records),
               "environment": environment()})
    return path
