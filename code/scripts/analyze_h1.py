"""Aggregate paired target effects within each fixed synthetic geometry."""
import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True, help='rectangles.csv')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
data = pd.read_csv(args.results)
paired = data.pivot(index=['seed', 'repeat', 'budget'], columns='target', values='gain')
if paired.isna().any().any():
    raise ValueError('Each geometry/repeat/budget must contain both targets')
paired['difference'] = paired['left-right'] - paired['up-down']
by_geometry = paired.groupby(['seed', 'budget']).difference.agg(['mean', 'size'])
if not by_geometry['size'].eq(20).all():
    raise ValueError('Expected twenty paired disclosures per geometry and budget')
summary = by_geometry.groupby('budget')['mean'].mean().to_frame('mean')
summary['size'] = paired.groupby('budget').size()
args.out.mkdir(parents=True, exist_ok=True)
paired.to_csv(args.out / 'h1_paired_effects.csv')
by_geometry.to_csv(args.out / 'h1_by_geometry.csv')
summary.to_csv(args.out / 'h1_paired_summary.csv')
print(summary.to_string())
