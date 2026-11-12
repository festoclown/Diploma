"""Restore the two principal result tables and original random seeds from CSV."""
import argparse
from pathlib import Path

import pandas as pd
from clusterization.storage import derived_seed

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=Path("reports/review_2026_11_12"))
parser.add_argument("--out", type=Path, default=Path("reports/main_tables"))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
draws = pd.read_csv(args.results / "draws.csv")
summary = pd.read_csv(args.results / "summary.csv")
names = {"breast_cancer": "Breast Cancer", "dry_bean": "Dry Bean", "digits": "Digits",
         "ag_news": "AG News", "20newsgroups": "20 Newsgroups", "banking77": "Banking77"}


def write_table(filename, headers, rows):
    lines = [r"\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}",
             r"\toprule", " & ".join(headers) + r"\\", r"\midrule"]
    for row in rows:
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append("$" + f"{value:.4f}".replace(".", "{,}") + "$")
            else:
                formatted.append(str(value))
        lines.append(" & ".join(formatted) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (args.out / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


absolute = []
baselines = []
seeds = []
for dataset in sorted(names):
    for budget in [20, 50]:
        block = draws[(draws.dataset == dataset) & (draws.budget == budget)]
        saved = summary[(summary.dataset == dataset) & (summary.budget == budget)].iloc[0]
        assert abs(block.M1_ari.mean() - saved.M1) < 1e-12
        assert abs(block.M6_ari.mean() - saved.M6) < 1e-12
        absolute.append([names[dataset], budget, saved.M1, saved.M6, saved.oracle,
                         saved.R1, saved.R6])
        baselines.append([names[dataset], budget] +
                         [block[m + "_ari"].mean() for m in ["M0", "M1", "M2", "M5", "M6", "M7"]])
        for repeat in block["repeat"]:
            seeds.append(dict(dataset=dataset, budget=budget, repeat=int(repeat),
                annotation_seed=derived_seed(dataset, int(repeat)),
                bootstrap_seed=derived_seed(dataset, int(repeat), budget), B=100))
write_table("absolute.tex", ["Набор", "$m$", "$Q_1$", "$Q_6$", r"$Q^\star$", "$R_1$", "$R_6$"],
            absolute)
write_table("baselines.tex", ["Набор", "$m$", "M0", "M1", "M2", "M5", "M6", "M7"], baselines)
pd.DataFrame(seeds).to_csv(args.out / "seeds.csv", index=False)
print("Restored absolute.tex, baselines.tex and seeds.csv in", args.out)
