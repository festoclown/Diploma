"""Paired selection experiment with held-out labels revealed only after selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backend import Backend
from .candidates import internal_scores, load_pool
from .config import dataset_key, validate
from .data import prepared_dir, verify_prepared
from .metrics import ari_grid, external_scores, positive_pairs
from .sampling import InapplicableScenario, annotation_order, bootstrap_weights, reserve_test_split
from .selection import choose
from .storage import (derived_seed, digest, environment, file_hash, new_directory,
                      read_json, write_json, write_jsonl, write_npz)

LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def tuning_contract(c: dict) -> dict:
    s = c["selection"]
    # PCA is a corpus-specific representation step, not a selector hyperparameter.
    candidate_settings = {k: v for k, v in c["candidates"].items()
                          if k != "projection_dim"}
    return {"candidates": candidate_settings, "compute": c["compute"],
            "internal": s["internal"],
            "budgets": sorted(s["budgets"]), "scenarios": sorted(s["scenarios"]),
            "bias_weight": s["bias_weight"], "fallback": s["fallback"],
            "evaluation": c["evaluation"], "lambda_grid": LAMBDA_GRID}


def frozen_weight(root: Path, c: dict) -> tuple[float | None, dict | None]:
    if "M4" not in c["selection"]["methods"]:
        return None, None
    suffix = "" if c["compute"]["backend"] == "cpu" else "_cuda"
    artifact = read_json(root / "artifacts" / f"frozen_mixture{suffix}.json")
    # M4 may use only a weight frozen under the same selection protocol;
    # changing budgets or candidate settings would invalidate that transfer.
    if artifact["contract"] != tuning_contract(c):
        raise ValueError("Frozen M4 protocol differs from this config")
    if not 0 <= artifact["lambda"] <= 1:
        raise ValueError("Invalid frozen lambda")
    return artifact["lambda"], artifact


def run(root: Path, c: dict, *, tuning: bool = False) -> Path:
    validate(c)
    if tuning and (c["phase"] != "development" or c["dataset"]["kind"] != "synthetic"):
        raise ValueError("Tuning is allowed only on development synthetic datasets")
    if not tuning and c["phase"] == "development":
        raise ValueError("Use tune_mixture.py for development configs")
    spec, s, evaluation = c["dataset"], c["selection"], c["evaluation"]
    backend = Backend(c["compute"]["backend"])
    prepared = prepared_dir(root, c)
    data_manifest = verify_prepared(prepared)
    partitions, pool_manifest = load_pool(root, c)
    raw, scaled = internal_scores(pool_manifest, s["internal"])
    weight, frozen = (None, None) if tuning else frozen_weight(root, c)
    # The result key includes the exact prepared data and candidate pool, so
    # a later source or parameter change cannot silently reuse old decisions.
    signature = digest({"config": c, "data": data_manifest, "pool": pool_manifest,
                        "tuning": tuning, "frozen": frozen})
    directory = root / "results" / c["name"] / dataset_key(c) / signature[:16]
    if directory.exists():
        manifest = read_json(directory / "manifest.json")
        if manifest.get("status") == "complete" and manifest["signature"] == signature:
            for name, checksum in manifest["files"].items():
                if file_hash(directory / name) != checksum:
                    raise ValueError(f"Result file changed: {name}")
            return directory
        raise ValueError(f"Incomplete run; inspect or move it before retry: {directory}")
    with np.load(prepared / "labels.npz", allow_pickle=False) as f:
        y, groups = f["y"], f["groups"]
    if partitions.shape[1] != len(y):
        raise ValueError("Pool and document count differ")
    # Candidate partitions already cover every object, but only reserve labels
    # can enter selectors. Test labels remain hidden until decisions are fixed.
    reserve, test = reserve_test_split(groups, evaluation["split_seed"], evaluation["test_fraction"])
    new_directory(directory)
    manifest = {"status": "running", "signature": signature, "config": c,
                "dataset_id": dataset_key(c), "pool": pool_manifest,
                "data_manifest": data_manifest, "frozen_mixture": frozen,
                "environment": environment(), "tuning": tuning}
    write_json(directory / "manifest.json", manifest)
    write_npz(directory / "split.npz", reserve=reserve, test=test)
    results, diagnostics, skipped = [], [], []
    test_scores = reserve_scores = None
    candidate_ids = [r["id"] for r in pool_manifest["candidates"]]
    for scenario in s["scenarios"]:
        for repeat in range(s["repeats"]):
            repeat_seed = s["repeat_seed_start"] + repeat
            draw_seed = derived_seed(dataset_key(c), scenario, repeat_seed, "annotations")
            try:
                order, draw_metadata = annotation_order(y[reserve], scenario, max(s["budgets"]),
                    np.random.default_rng(draw_seed), s["bias_weight"])
            except InapplicableScenario as exc:
                skipped.append({"scenario": scenario, "repeat": repeat, "reason": str(exc)})
                continue
            for budget in sorted(s["budgets"]):
                labelled = reserve[order[:budget]]
                partial = ari_grid(y[labelled], partitions[:, labelled], backend=backend)[0]
                boot_seed = derived_seed(draw_seed, budget, "bootstrap")
                # Recompute ARI on resampled labelled objects; clustering and
                # internal indices stay fixed across annotation repeats.
                weights = bootstrap_weights(groups[labelled], s["bootstrap_draws"],
                                            np.random.default_rng(boot_seed))
                bootstrap = ari_grid(y[labelled], partitions[:, labelled], weights,
                                     backend=backend)
                no_information = len(np.unique(y[labelled])) < 2 or positive_pairs(y[labelled]) == 0
                selectors = [("M4", a, f"lambda_{a:g}") for a in LAMBDA_GRID] if tuning else [
                    (m, weight, m) for m in s["methods"]]
                decisions = []
                for method, chosen_weight, method_name in selectors:
                    fallback = bool(s["fallback"] and no_information and method != "M0")
                    decision = choose("M0" if fallback else method, partial, raw, scaled, bootstrap,
                        fixed_window=s["fixed_window"], adaptive_multiplier=s["adaptive_multiplier"],
                        tuned_lambda=chosen_weight)
                    decisions.append((method_name, chosen_weight, fallback, decision))

                # Evaluation begins after every selector has committed its decision.
                if test_scores is None:
                    test_scores = external_scores(y[test], partitions[:, test], backend)
                    reserve_scores = ari_grid(y[reserve], partitions[:, reserve],
                                              backend=backend)[0]
                oracle = max(row["ari"] for row in test_scores)
                key = f"{scenario}-r{repeat:03d}-m{budget}"
                write_npz(directory / "draws" / f"{key}.npz", labelled=labelled,
                          partial_ari=partial, bootstrap_ari=bootstrap)
                # Coverage is retrospective diagnostics. It is deliberately
                # not passed into choose(), including for M6.
                observed_classes = np.unique(y[labelled])
                missing_test = test[~np.isin(y[test], observed_classes)]
                missing_ari = None
                if len(missing_test) >= 2 and len(np.unique(y[missing_test])) >= 2:
                    missing_ari = ari_grid(y[missing_test], partitions[:, missing_test],
                                           backend=backend)[0]
                class_counts = np.stack([weights[:, y[labelled] == k].sum(axis=1)
                                         for k in observed_classes], axis=1)
                bootstrap_degenerate = ((class_counts > 0).sum(axis=1) < 2) | (
                    (class_counts * (class_counts - 1) / 2).sum(axis=1) == 0)
                base = {"dataset": dataset_key(c), "kind": spec["kind"],
                    "cell": (spec["geometry"] + "_" + spec["balance"]) if spec["kind"] == "synthetic" else spec["name"],
                    "backend": backend.name,
                    "scenario": scenario, "budget": budget, "repeat": repeat,
                    "draw_seed": draw_seed, "bootstrap_seed": boot_seed,
                    "coverage": float(len(observed_classes) / len(np.unique(y))),
                    "positive_pairs": positive_pairs(y[labelled]),
                    "labelled_groups": int(len(np.unique(groups[labelled]))),
                    "no_information": no_information,
                    "bootstrap_degenerate_fraction": float(bootstrap_degenerate.mean()),
                    "test_n": len(test), "test_classes": int(len(np.unique(y[test]))),
                    "missing_test_n": len(missing_test), "oracle_ari": oracle,
                    "internal": s["internal"], "fixed_window": s["fixed_window"],
                    "adaptive_multiplier": s["adaptive_multiplier"], "fallback_enabled": s["fallback"]}
                for method_name, chosen_weight, fallback, decision in decisions:
                    selected = decision["selected"]
                    quality = test_scores[selected]
                    regret = max(0.0, oracle - quality["ari"])
                    results.append({**base, **decision, "method": method_name,
                        "tuned_lambda": chosen_weight if method_name == "M4" or tuning else None,
                        "fallback_used": fallback, "selected_id": candidate_ids[selected],
                        "selected_k": pool_manifest["candidates"][selected]["k"],
                        "selected_coverage": pool_manifest["candidates"][selected]["coverage"],
                        "selected_aliases": pool_manifest["candidates"][selected]["aliases"],
                        **{f"test_{k}": v for k, v in quality.items()}, "regret": regret,
                        "near_optimal": regret <= evaluation["near_optimal_delta"] + 1e-12,
                        "large_loss": regret > evaluation["large_loss_threshold"],
                        "missing_classes_ari": None if missing_ari is None else float(missing_ari[selected])})
                leader = decisions[0][3]["external_leader"]
                gap_sd = (bootstrap[:, leader, None] - bootstrap).std(axis=0, ddof=1)
                for j, identifier in enumerate(candidate_ids):
                    diagnostics.append({**base, "candidate_id": identifier,
                        "candidate_coverage": pool_manifest["candidates"][j]["coverage"],
                        "partial_ari": float(partial[j]), "reserve_ari": float(reserve_scores[j]),
                        "partial_minus_reserve": float(partial[j] - reserve_scores[j]),
                        "bootstrap_sd": float(bootstrap[:, j].std(ddof=1)),
                        "gap_to_leader": float(partial[leader] - partial[j]),
                        "gap_sd": float(gap_sd[j]), "draw_metadata": draw_metadata})
    write_jsonl(directory / "selections.jsonl", results)
    write_jsonl(directory / "diagnostics.jsonl", diagnostics)
    write_jsonl(directory / "skipped.jsonl", skipped)
    manifest.update(status="complete", rows=len(results), skipped=len(skipped),
                    files={str(p.relative_to(directory)): file_hash(p)
                           for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    write_json(directory / "manifest.json", manifest)
    return directory
