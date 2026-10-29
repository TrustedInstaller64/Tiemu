"""Remote paired baseline/candidate measurements; sequential, bounded and explicit."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
script = Path(__file__).with_name('benchmark_single_world.py').resolve()
configurations = [('baseline', args.baseline.resolve(), 1, 4096),
                  ('a-1', args.candidate.resolve(), 1, 4096),
                  ('a-4', args.candidate.resolve(), 4, 4096),
                  ('a-4-forced', args.candidate.resolve(), 4, 1)]
cases = [('human', 1000, 6), ('human', 10000, 3), ('aeonic', 200, 2)]
summary = []
for phase, population, generations in cases:
    results = {name: [] for name, _, _, _ in configurations}
    for repeat in range(3):
        # Rotate execution order to avoid always giving one build the first run.
        order = configurations[repeat:] + configurations[:repeat]
        for name, root, workers, threshold in order:
            output = args.output_dir/f'{phase}-{population}-{name}-{repeat}.json'
            command = [sys.executable, str(script), '--source-root', str(root), '--phase', phase,
                       '--population', str(population), '--generations', str(generations),
                       '--native-threads', str(workers), '--native-min-rows', str(threshold),
                       '--repeats', '1', '--output', str(output)]
            subprocess.run(command, check=True, timeout=180,
                           env={**os.environ, 'PYTHONHASHSEED': '0'}, stdout=subprocess.DEVNULL)
            results[name].append(json.loads(output.read_text(encoding='utf-8')))
    medians = {name: statistics.median(run['median_seconds'] for run in runs)
               for name, runs in results.items()}
    digests = {run['runs'][0]['state_digest'] for runs in results.values() for run in runs}
    row = {'phase': phase, 'population': population, 'generations': generations,
           'median_seconds': medians,
           'speedup_over_baseline': {name: medians['baseline']/value for name, value in medians.items()},
           'same_checked_state_and_rng_digest': len(digests) == 1,
           'mean_cpu_cores': {name: statistics.median(r['runs'][0]['cpu_seconds']/r['runs'][0]['seconds']
                                                    for r in runs) for name, runs in results.items()}}
    summary.append(row)
    print(json.dumps(row), flush=True)
report = {'baseline': str(args.baseline.resolve()), 'candidate': str(args.candidate.resolve()),
          'scope': 'Same runner, sequential isolated processes, 3 rotated repeats. '
                   'No LLM/log/display/exports; actual online training only in aeonic case. '
                   'Digest covers recorded entities, ordering, blueprint, seats, network weights, '
                   'Adam state, and Python/NumPy/Torch RNG, not every internal manager field. '
                   'Not a full-world or local-M1 performance claim.', 'cases': summary}
(args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
if any(not row['same_checked_state_and_rng_digest'] for row in summary):
    raise SystemExit('Checked state differs: inspect the regression evidence before accepting speedups')
