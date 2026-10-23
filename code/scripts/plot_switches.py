"""Plot saved diagnostic CSV independently of computation."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument("--report", required=True)
a = p.parse_args()
root = Path(a.report)
f = pd.read_csv(root / "switches.csv")
for i, (key, part) in enumerate(f[f.method == "M6"].groupby(["run", "scenario", "budget"])):
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in part.itertuples():
        ax.plot([0, 1], [row.test_ari_M1, row.test_ari], marker="o", alpha=.7)
    ax.set_xticks([0, 1], ["External leader (M1)", "Adaptive choice (M6)"])
    ax.set_ylabel("Held-out ARI")
    ax.set_title(f"{part.dataset.iloc[0]} | {key[1]} | m={key[2]} | n={len(part)}")
    fig.tight_layout()
    fig.savefig(root / f"switches-{i:03d}.png", dpi=160)
    plt.close(fig)
