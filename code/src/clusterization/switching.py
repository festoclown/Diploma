"""Paired diagnostics and exploratory threshold sweeps on fixed saved draws."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .selection import choose
from .candidates import load_pool, internal_scores
from .data import prepared_dir
from .metrics import external_scores
from .backend import Backend


def analyze(source, out):
    source, out = Path(source), Path(out)
    records = []
    for path in sorted(source.rglob("selections.jsonl")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete run: {path.parent}")
        if manifest.get("tuning"):
            continue
        rows = pd.read_json(path, lines=True)
        if rows.empty:
            continue
        keys = ["dataset", "scenario", "budget", "repeat"]
        if rows.duplicated(keys + ["method"]).any():
            raise ValueError(f"Duplicate observations: {path}")
        base = rows[rows.method == "M1"][keys + ["test_ari", "selected_id"]]
        paired = rows.merge(base, on=keys, suffixes=("", "_M1"), validate="many_to_one")
        if len(paired) != len(rows):
            raise ValueError(f"Missing M1 pairs: {path}")
        paired["gain"] = paired.test_ari - paired.test_ari_M1
        paired["switched"] = paired.selected_id != paired.selected_id_M1
        paired["win"] = paired.gain > 1e-12
        paired["loss"] = paired.gain < -1e-12
        paired["run"] = str(path.parent.relative_to(source))
        records.append(paired)
    if not records:
        raise ValueError("No complete runs with saved decisions")
    frame = pd.concat(records, ignore_index=True)
    keys = ["run", "dataset", "scenario", "budget", "method"]
    summary = frame.groupby(keys).agg(n=("gain", "size"), mean_gain=("gain", "mean"),
        switch_rate=("switched", "mean"), win_rate=("win", "mean"), loss_rate=("loss", "mean"),
        worst_gain=("gain", "min"), mean_regret=("regret", "mean")).reset_index()
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "switches.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    return frame, summary


def sweep(root, run, out):
    """Vary thresholds on identical saved labels. Test labels only score decisions."""
    root, run, out = Path(root), Path(run), Path(out)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Run must be complete")
    config = manifest["config"]
    if config["selection"]["fallback"]:
        raise ValueError("This sweep requires a run without fallback")
    partitions, pool = load_pool(root, config)
    raw, scaled = internal_scores(pool, config["selection"]["internal"])
    with np.load(prepared_dir(root, config) / "labels.npz") as data:
        y = data["y"]
    with np.load(run / "split.npz") as data:
        test = data["test"]
    quality = np.array([x["ari"] for x in external_scores(y[test], partitions[:, test], Backend("cpu"))])
    records = []
    for path in sorted((run / "draws").glob("*.npz")):
        with np.load(path) as data:
            partial, boot = data["partial_ari"], data["bootstrap_ari"]
        leader = choose("M1", partial, raw, scaled, boot)["selected"]
        for method, values in [("M6", [0., .5, 1., 2.]), ("M5", [0., .02, .05, .1])]:
            for value in values:
                kwargs = {"adaptive_multiplier" if method == "M6" else "fixed_window": value}
                decision = choose(method, partial, raw, scaled, boot, **kwargs)
                selected = decision["selected"]
                records.append(dict(draw=path.stem, method=method, threshold=value,
                    selected_id=pool["candidates"][selected]["id"],
                    gain=float(quality[selected]-quality[leader]),
                    regret=float(quality.max()-quality[selected]), **decision))
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No saved bootstrap draws")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame
