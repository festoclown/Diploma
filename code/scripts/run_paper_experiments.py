"""Recompute all experiments reported in the paper from repository-local inputs.

Run from an installed checkout with --out pointing to a results directory.
No network access or embedding model is needed for the frozen-input protocol.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import pandas as pd

from clusterization.light_series import archived, rectangles
from clusterization.storage import environment

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
(out / "run_manifest.json").write_text(json.dumps({"status": "incomplete"}))
inputs = out / 'inputs'
inputs.mkdir(exist_ok=True)
fixture = project / 'fixtures/paper_inputs.zip'
with ZipFile(fixture) as archive:
    manifest = json.loads(archive.read('input_manifest.json'))
    for name, checksum in manifest.items():
        target = (inputs / name).resolve()
        if not target.is_relative_to(inputs.resolve()):
            raise ValueError(f'Input path outside the output directory: {name}')
        content = archive.read(name)
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f'Input checksum mismatch: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

stages = []
started = perf_counter()


def run_script(filename, *arguments):
    before = perf_counter()
    command = [sys.executable, str(project / 'scripts' / filename), *map(str, arguments)]
    print('Running', filename, flush=True)
    subprocess.run(command, cwd=project, check=True)
    stages.append(dict(stage=filename, seconds=perf_counter() - before))


diagnostics = out / 'diagnostics'
refinement = out / 'refinement'
main = out / 'main'
main.mkdir(exist_ok=True)
run_script('review_analysis.py', '--root', inputs, '--archive', inputs / 'pilot_replay',
           '--out', diagnostics)
# Export fresh decisions for all six domain tasks, then recompute both main controls.
draws = pd.read_csv(diagnostics / 'draws.csv')
for name, block in draws.groupby('dataset'):
    block.to_csv(main / (name + '.csv'), index=False)
archived(inputs / 'pilot_replay', 'cpu').to_csv(main / 'pilots.csv', index=False)
rectangles('cpu').to_csv(main / 'rectangles.csv', index=False)
archived(inputs / 'pilot_replay', 'cpu', stability=True).to_csv(
    main / 'bootstrap.csv', index=False)
run_script('refine_review_analysis.py', '--results', diagnostics, '--out', refinement)
run_script('build_main_tables.py', '--results', diagnostics, '--out', out / 'tables')
run_script('analyze_h1.py', '--results', main / 'rectangles.csv', '--out', out / 'tables')
run_script('build_experiment_figures.py', '--results', main, '--diagnostics', diagnostics,
           '--refinement', refinement, '--out', out / 'figures')
record = dict(status='complete', elapsed_seconds=perf_counter() - started, stages=stages,
    input_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(), environment=environment(),
    decisions=len(draws), input_files=len(manifest))
(out / 'run_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(f'Completed all paper experiments in {record["elapsed_seconds"]:.1f} s: {out}')
