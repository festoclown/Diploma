"""Analyze completed runs without rerunning clustering."""
import argparse
from clusterization.switching import analyze
p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
frame, summary = analyze(a.results, a.out)
print(summary.to_string(index=False))
