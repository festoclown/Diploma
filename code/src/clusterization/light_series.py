"""Four paired experiments. The same selection rules consume CPU or CUDA ARI grids."""
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .backend import Backend
from .candidates import _partition, canonical, internal_scores, load_pool
from .data import prepare, prepared_dir
from .downloads import download_data, download_model
from .embeddings import encode, vector_path
from .metrics import ari_grid
from .sampling import bootstrap_weights, reserve_test_split
from .selection import TOLERANCE, choose
from .storage import derived_seed, digest, read_json, read_jsonl

DOMAINS = {"breast_cancer": "измерения", "digits": "изображения", "dry_bean": "измерения",
           "ag_news": "тексты", "20newsgroups": "тексты", "banking77": "тексты"}
PILOTS = ("gpu_pilot", "gpu_breast_cancer_light", "gpu_digits_light", "gpu_digits_db_light")


def evaluate(partial, boot, raw, quality, candidates):
    """Test scores only enter diagnostics after all four choices have been fixed."""
    choices = {m: choose(m, partial, raw, raw, boot)["selected"] for m in ("M1","M2","M5","M6")}
    leader, selected = choices["M1"], choices["M6"]
    spread = (boot[:, leader, None] - boot).std(axis=0, ddof=1)
    margin = spread - (partial[leader] - partial)
    allowed = margin >= -TOLERANCE
    best = quality[allowed].max()
    row = {"pool_size": len(raw), "pool_min": quality.min(), "oracle": quality.max(),
           "eligible": int(allowed.sum()), "excluded_loss": quality.max()-best,
           "inside_loss": best-quality[selected], "switched": selected != leader,
           "gain": quality[selected]-quality[leader]}
    for method, index in choices.items():
        row[method] = candidates[index]["id"]
        row[method+"_ari"] = quality[index]
        row[method+"_candidate"] = candidates[index]["aliases"]
        row[method+"_k"] = candidates[index]["k"]
    return row, margin


def archive_inputs(archive, name):
    runs = sorted((archive / "results" / name).glob("*/*/manifest.json"))
    if len(runs) != 1:
        raise ValueError(f"Expected one archived run for {name} under {archive}, found {len(runs)}")
    directory = runs[0].parent
    manifest = read_json(runs[0])
    config = manifest["config"]
    pool, info = load_pool(archive, config)
    if info["sha256"] != manifest["pool"]["sha256"]:
        raise ValueError("Archived partitions differ from the run")
    raw, _ = internal_scores(info, config["selection"]["internal"])
    with np.load(prepared_dir(archive, config) / "labels.npz") as f:
        y, groups = f["y"], f["groups"]
    with np.load(directory / "split.npz") as f:
        test = f["test"]
    saved = pd.DataFrame(read_jsonl(directory / "selections.jsonl"))
    return directory, pool, info["candidates"], raw, y, groups, test, saved


def archived(archive, device, stability=False):
    backend = Backend(device)
    output = []
    names = PILOTS[1:3] if stability else PILOTS
    for name in names:
        directory, pool, candidates, raw, y, groups, test, saved = archive_inputs(archive, name)
        quality = ari_grid(y[test], pool[:,test], backend=backend)[0]
        seen = set()
        for (scenario, budget, repeat), block in saved.groupby(["scenario","budget","repeat"]):
            saved_ids = block.set_index("method").selected_id
            changed = saved_ids.M1 != saved_ids.M6
            if stability and (scenario != "random" or changed in seen):
                continue
            seen.add(changed)
            key = f"{scenario}-r{repeat:03d}-m{budget}"
            with np.load(directory / "draws" / (key+".npz")) as f:
                labelled, partial, original = f["labelled"], f["partial_ari"], f["bootstrap_ari"]
            base = dict(pilot=name, scenario=scenario, budget=budget, repeat=repeat,
                        archived_switch=changed)
            observed = ari_grid(y[labelled], pool[:,labelled], backend=backend)[0]
            np.testing.assert_allclose(partial, observed, atol=1e-10)
            if not stability:
                row, _ = evaluate(partial, original, raw, quality, candidates)
                for method in ("M1","M2","M5","M6"):
                    if row[method] != saved_ids[method]:
                        raise ValueError(f"Decision mismatch: {name}/{key}/{method}")
                output.append({**base, **row})
                continue
            for draws in (30,100,300):
                for seed in range(5):
                    weights = bootstrap_weights(groups[labelled], draws, np.random.default_rng(seed))
                    boot = ari_grid(y[labelled], pool[:,labelled], weights, backend=backend)
                    row, margin = evaluate(partial, boot, raw, quality, candidates)
                    alternatives = [i for i,c in enumerate(candidates) if c["id"] != row["M1"]]
                    nearest = min(alternatives, key=lambda i: abs(margin[i])) if alternatives else None
                    output.append({**base, **row, "draws": draws, "seed": seed,
                        "same_as_archive": row["M6"] == saved_ids.M6,
                        "nearest_id": candidates[nearest]["id"] if nearest is not None else None,
                        "margin": margin[nearest] if nearest is not None else np.nan})
    return pd.DataFrame(output)


def make_pool(backend, x, rectangle=False):
    params = dict(n_init=20, max_iter=300, seed=42, spectral_neighbors=15, spectral_n_init=10)
    grid = [("kmeans", k) for k in ((2,3,4) if rectangle else (2,4,8,16))]
    if not rectangle:
        grid += [("spectral",4), ("spectral",8)]
    distance = backend.distances(x, "euclidean")
    partitions = [(f"{algo}-{k}", _partition(backend,x,algo,k,params)) for algo,k in grid]
    if rectangle:
        partitions.extend((name, backend.host(x[:,j] > 0)) for j,name in enumerate(("left-right","up-down")))
    unique = {}
    for alias, labels in partitions:
        labels = canonical(labels)
        key = digest(labels.tolist())
        if key not in unique:
            score, _, _ = backend.internal(x,distance,labels)
            unique[key] = (labels, {"id":key, "k":len(np.unique(labels)), "aliases":[],
                                    "silhouette":score})
        unique[key][1]["aliases"].append(alias)
    ordered = [unique[k] for k in sorted(unique)]
    return np.stack([p for p,_ in ordered]), [c for _,c in ordered]


def trials(backend, pool, candidates, y, groups, budgets, sample_seed, axes=None):
    reserve, test = reserve_test_split(groups, 31415)
    if len(reserve) < max(budgets):
        raise ValueError("Annotation budget exceeds reserve")
    quality = ari_grid(y[test], pool[:,test], backend=backend)[0]
    raw = np.array([c["silhouette"] for c in candidates])
    for repeat in range(20):
        order = np.random.default_rng(derived_seed(sample_seed,repeat)).permutation(reserve)
        for budget in budgets:
            idx = order[:budget]
            weights = bootstrap_weights(groups[idx],100,np.random.default_rng(
                derived_seed(sample_seed,repeat,budget)))
            partial = ari_grid(y[idx],pool[:,idx],backend=backend)[0]
            boot = ari_grid(y[idx],pool[:,idx],weights,backend=backend)
            row, _ = evaluate(partial,boot,raw,quality,candidates)
            row.update(repeat=repeat,budget=budget,coverage=len(np.unique(y[idx]))/len(np.unique(y)))
            if axes is not None:
                row["controls_tied_on_labels"] = np.array_equal(canonical(axes[0][idx]),canonical(axes[1][idx]))
                row["both_classes"] = len(np.unique(y[idx])) == 2
            yield row


def rectangles(device):
    backend = Backend(device)
    rows = []
    for seed in (101,102,103):
        centers = np.repeat([[-3,-1],[-3,1],[3,-1],[3,1]],100,axis=0)
        x = (centers + np.random.default_rng(seed).normal(0,.2,centers.shape)).astype("float32")
        pool, candidates = make_pool(backend,backend.array(x),rectangle=True)
        axes = (x[:,0]>0,x[:,1]>0)
        for axis, target in enumerate(("left-right","up-down")):
            silhouette = next(c["silhouette"] for c in candidates if target in c["aliases"])
            for row in trials(backend,pool,candidates,axes[axis],np.arange(len(x)),(4,8,20),seed,axes):
                rows.append({**row,"seed":seed,"target":target,"control_silhouette":silhouette})
    return pd.DataFrame(rows)


def domain(root, name, device):
    backend = Backend(device)
    text = DOMAINS[name] == "тексты"
    spec = dict(kind="text" if text else "tabular",name=name,subsample_seed=2026)
    spec.update(dict(max_documents=600,source_revision="main") if text else dict(max_samples=600))
    config = {"dataset":spec, "model":dict(repo="sentence-transformers/all-MiniLM-L6-v2",
              revision="main",max_length=256,batch_size=64,device=device)}
    start = perf_counter()
    download_data(root,spec)
    if text:
        download_model(root,config["model"])
    data = prepare(root,config)
    prepared = perf_counter()
    if text:
        encode(root,config)
    with np.load(vector_path(root,config)) as f:
        x = backend.array(f["x"])
    if text:
        x = backend.project(x,50)
    encoded = perf_counter()
    pool,candidates = make_pool(backend,x)
    clustered = perf_counter()
    with np.load(data/"labels.npz") as f:
        rows = list(trials(backend,pool,candidates,f["y"],f["groups"],(20,50),name))
    result = pd.DataFrame(rows).assign(dataset=name,domain=DOMAINS[name])
    timings = dict(dataset=name,prepare_s=prepared-start,encode_project_s=encoded-prepared,
                   cluster_s=clustered-encoded,selection_s=perf_counter()-clustered)
    return result,timings
