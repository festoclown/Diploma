"""Descriptive reports and paired intervals at explicit units of replication."""

from pathlib import Path

import numpy as np
import pandas as pd

from .storage import derived_seed, digest, new_directory, read_json, read_jsonl, write_json


def mean_interval(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    # These are descriptive percentile intervals over the units supplied by
    # the caller. Annotation repeats and independent datasets are not the
    # same sampling unit and are summarized separately below.
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(source: Path, target: Path) -> None:
    rows, manifests = [], []
    for path in sorted(source.rglob("manifest.json")):
        manifest = read_json(path)
        if "config" not in manifest or "status" not in manifest:
            continue
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete run under report source: {path.parent}")
        if manifest.get("tuning"):
            continue
        c = manifest["config"]
        series = digest({key: c[key] for key in ["selection", "candidates", "evaluation", "model", "phase"]})[:16]
        for row in read_jsonl(path.parent / "selections.jsonl"):
            rows.append({**row, "series": series, "phase": c["phase"]})
        manifests.append({"run": str(path.parent), "signature": manifest["signature"],
                          "series": series, "config": c, "skipped": manifest["skipped"]})
    if not rows:
        raise ValueError("No completed non-development results found")
    frame = pd.DataFrame(rows)
    # Reject accidental mixing of duplicate runs before any averaging; it
    # would otherwise make the apparent sample size too large.
    identity = ["series", "dataset", "scenario", "budget", "repeat", "method"]
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate observations across runs; report one source/revision at a time")
    new_directory(target)
    frame.to_csv(target / "selections.csv", index=False)
    group = ["series", "phase", "kind", "cell", "dataset", "scenario", "budget", "method"]
    summary = frame.groupby(group, dropna=False).agg(
        mean_regret=("regret", "mean"), median_regret=("regret", "median"),
        mean_ari=("test_ari", "mean"), mean_ami=("test_ami", "mean"),
        near_optimal_rate=("near_optimal", "mean"), large_loss_rate=("large_loss", "mean"),
        mean_coverage=("coverage", "mean"), mean_eligible=("eligible_count", "mean"),
        fallback_rate=("fallback_used", "mean"), repeats=("repeat", "nunique")).reset_index()
    summary.to_csv(target / "per_dataset.csv", index=False)
    contrasts = []
    keys = ["series", "phase", "kind", "cell", "dataset", "scenario", "budget"]
    for values, sub in frame.groupby(keys, dropna=False):
        # Pair methods by the same annotation repeat on the same fixed corpus.
        # Positive baseline - M6 means M6 has smaller external regret.
        pivot = sub.pivot(index="repeat", columns="method", values="regret")
        if "M6" not in pivot:
            continue
        for baseline in ["M0", "M1", "M2", "M3", "M4", "M5", "M7"]:
            if baseline not in pivot:
                continue
            diff = (pivot[baseline] - pivot["M6"]).dropna().to_numpy()
            lo, hi = mean_interval(diff, derived_seed(list(values), baseline, "paired_ci"))
            contrasts.append({**dict(zip(keys, values)), "baseline": baseline,
                "mean_gain_M6": float(diff.mean()), "median_gain_M6": float(np.median(diff)),
                "ci_low": lo, "ci_high": hi, "n_repeats": len(diff),
                "unit": "annotation_repeat_on_fixed_corpus"})
    paired = pd.DataFrame(contrasts)
    paired.to_csv(target / "paired_contrasts.csv", index=False)
    synthetic = []
    if len(paired):
        keys2 = ["series", "phase", "cell", "scenario", "budget", "baseline"]
        for values, sub in paired[paired.kind == "synthetic"].groupby(keys2):
            # Aggregate repeats within each synthetic realization first; only
            # then resample realizations for a cross-dataset interval.
            gains = sub["mean_gain_M6"].to_numpy()
            lo, hi = mean_interval(gains, derived_seed(list(values), "realization_ci"))
            synthetic.append({**dict(zip(keys2, values)), "mean_gain_M6": float(gains.mean()),
                              "ci_low": lo, "ci_high": hi, "n_realizations": len(gains),
                              "unit": "synthetic_realization"})
    pd.DataFrame(synthetic).to_csv(target / "synthetic_contrasts.csv", index=False)
    text = summary[summary.kind == "text"]
    text.groupby(["series", "scenario", "budget", "method"])[
        ["mean_regret", "mean_ari", "near_optimal_rate", "large_loss_rate"]
    ].mean().to_csv(target / "text_macro_descriptive.csv")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for values, sub in summary.groupby(["series", "kind", "cell", "scenario"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, part in sub.groupby("method"):
            means = part.groupby("budget")["mean_regret"].mean()
            ax.plot(means.index, means.values, marker="o", label=method)
        ax.set(xlabel="Label budget", ylabel="Mean external regret", title=" / ".join(values[1:]))
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(target / ("regret-" + digest(list(values))[:16] + ".png"), dpi=180)
        plt.close(fig)
    write_json(target / "manifest.json", {"sources": manifests, "rows": len(frame),
               "interval": "percentile bootstrap of paired differences; 2000 draws",
               "p_values": "not computed; confirmatory significance testing requires a predefined family"})
