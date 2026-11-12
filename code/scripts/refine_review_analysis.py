"""Additional reporting for the second review, using only frozen arrays and CSV."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clusterization.metrics import ari_grid
from clusterization.sampling import bootstrap_weights
from clusterization.selection import choose, TOLERANCE
from clusterization.storage import derived_seed

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root, out = args.results, args.out
out.mkdir(parents=True, exist_ok=True)
draws = pd.read_csv(root / "draws.csv")
candidates = pd.read_csv(root / "candidates.csv")
frozen = root / "frozen"


def arrays(name):
    with np.load(frozen / (name + ".npz")) as data:
        return {k: data[k] for k in data.files}


def bootstrap(name, repeat, budget, B, seed):
    data = arrays(name)
    order = np.random.default_rng(derived_seed(name, repeat)).permutation(data["reserve"])
    idx = order[:budget]
    partial = ari_grid(data["y"][idx], data["pool"][:, idx])[0]
    weights = bootstrap_weights(data["groups"][idx], B, np.random.default_rng(seed))
    boot = ari_grid(data["y"][idx], data["pool"][:, idx], weights)
    cand = json.loads((frozen / (name + "-candidates.json")).read_text())
    raw = np.array([c["silhouette"] for c in cand])
    return data, cand, partial, boot, raw


def mean_interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(20261112)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975])


# Every comparison uses paired decisions from the same annotations.
comparisons = []
for (name, budget), group in draws.groupby(["dataset", "budget"]):
    for other in ["M0", "M7"]:
        effect = group.M6_ari - group[other + "_ari"]
        lo, hi = mean_interval(effect)
        comparisons.append(dict(dataset=name, budget=budget, baseline=other,
            mean=effect.mean(), low=lo, high=hi,
            different=int((group.M6 != group[other]).sum()),
            worse=int((effect < -TOLERANCE).sum()),
            better=int((effect > TOLERANCE).sum())))
pd.DataFrame(comparisons).to_csv(out / "paired_baselines.csv", index=False)

pool_rows = []
pool_source = pd.read_csv(root / "pool_control.csv")
for row in pool_source.to_dict("records"):
    name, repeat, budget = row["dataset"], row["repeat"], row["budget"]
    data, cand, partial, boot, raw = bootstrap(
        name, repeat, budget, 100, derived_seed(name, repeat, budget))
    mask = np.array([row["pool"] == "full" or any(a.startswith("kmeans") for a in c["aliases"])
                     for c in cand])
    chosen = choose("M6", partial[mask], raw[mask], raw[mask], boot[:, mask])
    q = ari_grid(data["y"][data["test"]], data["pool"][mask][:, data["test"]])[0]
    ids = [c["id"] for c, keep in zip(cand, mask) if keep]
    j1, j6 = ids.index(row["M1"]), ids.index(row["M6"])
    assert ids[chosen["selected"]] == row["M6"]
    pool_rows.append(dict(**row, Q1=q[j1], Q6=q[j6], oracle=q.max(),
        eligible=chosen["eligible_count"], switched=row["M1"] != row["M6"]))
pd.DataFrame(pool_rows).to_csv(out / "pool_absolute.csv", index=False)

coverage = pd.read_csv(root / "coverage_control.csv")
classes_rows, coverage_rows, coverage_summary = [], [], []
for name, group in coverage.groupby("dataset"):
    data = arrays(name)
    reserve, y = data["reserve"], data["y"]
    eligible = [int(k) for k in np.unique(y[reserve]) if (y[reserve] == k).sum() >= 10]
    for present, block in group.groupby("present"):
        for row in block.to_dict("records"):
            repeat = row["repeat"]
            rng = np.random.default_rng(derived_seed(name, "coverage", present, repeat))
            selected = rng.choice(eligible, present, replace=False)
            quotas = np.full(present, 20 // present)
            quotas[:20 % present] += 1
            idx = np.concatenate([rng.choice(reserve[y[reserve] == k], n, replace=False)
                                  for k, n in zip(selected, quotas)])
            coverage_rows.append(dict(**row, categories=json.dumps(selected.tolist()),
                labelled=json.dumps(idx.tolist()), quotas=json.dumps(quotas.tolist())))
            for k in eligible:
                classes_rows.append(dict(dataset=name, present=present, repeat=repeat,
                    category=k, selected=bool(k in selected),
                    reserve_n=int((y[reserve] == k).sum())))
        lo, hi = mean_interval(block.gain)
        coverage_summary.append(dict(dataset=name, present=present,
            mean=block.gain.mean(), low=lo, high=hi,
            switches=int(block.switched.sum()),
            worse=int((block.gain < -TOLERANCE).sum()),
            eligible_categories=json.dumps(eligible)))
pd.DataFrame(coverage_rows).to_csv(out / "coverage_draws.csv", index=False)
pd.DataFrame(classes_rows).to_csv(out / "coverage_categories.csv", index=False)
pd.DataFrame(coverage_summary).to_csv(out / "coverage_summary.csv", index=False)

references, reference_membership = [], []
for name in ["digits", "breast_cancer"]:
    for repeat in range(20):
        for budget in [20, 50]:
            for seed in [99, 100, 101, 102, 103]:
                data, cand, a, boot, raw = bootstrap(name, repeat, budget, 3000, seed)
                result = choose("M6", a, raw, raw, boot)
                q = ari_grid(data["y"][data["test"]], data["pool"][:, data["test"]])[0]
                j = result["selected"]
                leader = result["external_leader"]
                sd = (boot[:, leader, None] - boot).std(axis=0, ddof=1)
                references.append(dict(dataset=name, repeat=repeat, budget=budget,
                    seed=seed, M6=cand[j]["id"], Q6=q[j], gain=q[j] - q[leader]))
                for i, c in enumerate(cand):
                    reference_membership.append(dict(dataset=name, repeat=repeat, budget=budget,
                        seed=seed, candidate=c["id"],
                        margin=sd[i]-(a[leader]-a[i])+TOLERANCE))
refs = pd.DataFrame(references)
refs.to_csv(out / "bootstrap_references.csv", index=False)
pd.DataFrame(reference_membership).to_csv(out / "bootstrap_reference_membership.csv", index=False)
b = pd.read_csv(root / "bootstrap_control.csv")
ref = refs[refs.seed == 99].drop(columns="seed").rename(
    columns={"M6": "reference_id", "Q6": "reference_Q", "gain": "reference_G"})
b = b.merge(ref, on=["dataset", "repeat", "budget"], validate="many_to_one")
b["delta_Q"] = b.M6_ari - b.reference_Q
b["delta_G"] = b.gain - b.reference_G
assert np.array_equal(b.same_as_reference, b.M6 == b.reference_id)
b.to_csv(out / "bootstrap_map.csv", index=False)
# Check the prefix property of draws from the same seed.
data = arrays("digits")
idx = np.random.default_rng(derived_seed("digits", 7)).permutation(data["reserve"])[:20]
for seed in range(5):
    long = bootstrap_weights(data["groups"][idx], 300, np.random.default_rng(seed))
    for B in [30, 100]:
        short = bootstrap_weights(data["groups"][idx], B, np.random.default_rng(seed))
        assert np.array_equal(long[:B], short)
(out / "bootstrap_design.json").write_text(json.dumps(
    dict(nested=True, seeds=list(range(5)), reference_seeds=[99,100,101,102,103]), indent=2))
# H3 concerns information on T, without any assigned geometric preference.
example = json.loads((root / "H3_example.json").read_text())
y, partitions = np.array(example["y"]), np.array(example["partitions"])
example.update(reserve=[0,1,2,3], labelled=[0,1,2,3], test=[4,5],
    test_ari=ari_grid(y[4:], partitions[:,4:])[0].tolist(),
    scope="information ambiguity on held-out labels; no asserted M6 choice")
(out / "H3_example.json").write_text(json.dumps(example, indent=2))
print("Completed second-review diagnostics", out)
