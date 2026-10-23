"""Freeze exactly one mixture weight from synthetic development experiments."""

from pathlib import Path

import pandas as pd

from .experiment import run, tuning_contract
from .storage import digest, read_json, read_jsonl, write_json


def tune(root: Path, configs: list[dict]) -> Path:
    suffix = "" if configs and configs[0]["compute"]["backend"] == "cpu" else "_cuda"
    target = root / "artifacts" / f"frozen_mixture{suffix}.json"
    if target.exists():
        raise FileExistsError("Mixture is already frozen; never silently retune after confirmation")
    if not configs or any(c["phase"] != "development" or c["dataset"]["kind"] != "synthetic" for c in configs):
        raise ValueError("Only development synthetic configs are permitted")
    contract = tuning_contract(configs[0])
    if any(tuning_contract(c) != contract for c in configs):
        raise ValueError("All development configs must share one selection protocol")
    expected = {(g, b, seed) for g in ["separated", "overlapping", "elongated"]
                for b in ["balanced", "imbalanced"] for seed in [0, 1, 2]}
    actual = {(c["dataset"]["geometry"], c["dataset"]["balance"], c["dataset"]["seed"]) for c in configs}
    if actual != expected or len(configs) != 18:
        raise ValueError("M4 requires the full 18-dataset development grid")
    # Confirmation data must not influence lambda: only the 18 predefined
    # development datasets are run before the frozen artifact is written.
    paths = [run(root, c, tuning=True) for c in configs]
    if any(read_json(path / "manifest.json")["skipped"] for path in paths):
        raise ValueError("Development conditions were skipped; inspect before freezing M4")
    frame = pd.DataFrame([r for path in paths for r in read_jsonl(path / "selections.jsonl")])
    # Equal weight per condition, then per dataset; repeats are paired within a condition.
    per_condition = frame.groupby(["dataset", "scenario", "budget", "tuned_lambda"])["regret"].mean()
    per_dataset = per_condition.groupby(["dataset", "tuned_lambda"]).mean()
    scores = per_dataset.groupby("tuned_lambda").mean().sort_index()
    # The index is sorted above, so equal mean regret deterministically picks
    # the smallest lambda rather than depending on iteration order.
    best = float(scores[scores <= scores.min() + 1e-12].index[0])
    write_json(target, {"lambda": best, "contract": contract, "phase": "development",
               "criterion": "equal_dataset_mean_regret", "tie_break": "smallest_lambda",
               "scores": {str(k): float(v) for k, v in scores.items()},
               "development_config_hashes": [digest(c) for c in configs],
               "development_runs": [str(p.relative_to(root)) for p in paths]})
    return target
