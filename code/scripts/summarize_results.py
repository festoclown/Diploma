import argparse
from pathlib import Path
from clusterization.reporting import summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV summaries, paired intervals and PNG plots")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args()
    summarize(args.results.resolve(), args.out.resolve())
