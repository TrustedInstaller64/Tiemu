"""One remote inorganic scenario, paired across isolated source checkouts."""
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
script = Path(__file__).with_name('benchmark_simulation.py').resolve()
configurations = [('baseline', args.baseline.resolve()), ('candidate', args.candidate.resolve())]
generations = 10000
results = {name: [] for name, _ in configurations}
for repeat in range(3):
    # Alternate which checkout runs first; each subprocess owns one world.
    offset = repeat % len(configurations)
    order = configurations[offset:] + configurations[:offset]
    for name, root in order:
        output = args.output_dir / f'inorganic-10000-{name}-{repeat}.json'
        command = [sys.executable, str(script), '--source-root', str(root),
                   '--phase', 'inorganic', '--generations', str(generations),
                   '--backend', 'cpp', '--repeats', '1', '--output', str(output)]
        subprocess.run(command, check=True, timeout=180,
                       env={**os.environ, 'PYTHONHASHSEED': '0'}, stdout=subprocess.DEVNULL)
        results[name].append(json.loads(output.read_text(encoding='utf-8')))

medians = {name: statistics.median(run['median_seconds'] for run in runs)
           for name, runs in results.items()}
digests = {run['checked_states'][0]['state_digest'] for runs in results.values() for run in runs}
legacy_results = {json.dumps(run['outcomes'][0], sort_keys=True, allow_nan=False)
                  for runs in results.values() for run in runs}
row = {'phase': 'inorganic', 'grid': [10, 10], 'generations': generations,
       'median_seconds': medians,
       'speedup_over_baseline': {name: medians['baseline'] / value for name, value in medians.items()},
       'same_legacy': len(legacy_results) == 1,
       'same_checked_state_and_rng_digest': len(digests) == 1,
       'mean_cpu_cores': {name: statistics.median(run['cpu_seconds'][0] / run['seconds'][0]
                                                for run in runs) for name, runs in results.items()}}
report = {
    'baseline': str(args.baseline.resolve()), 'candidate': str(args.candidate.resolve()),
    'commits': {name: sorted({run['commit'] for run in runs}) for name, runs in results.items()},
    'scope': 'Hosted CPU runner only. Same runner, sequential isolated processes, 3 alternating '
             'repetitions of one 10000-generation 10x10 inorganic scenario. Uses the actual '
             '_run_inorganic_phase method with no LLM/log/display/keyboard/exports. Timings include '
             'grid initialization, RNG consumption, Python/native data movement and final legacy '
             'reduction, but exclude process/import/simulation setup, warm-up, state hashing and '
             'close. The candidate headless C ABI path checks controls every 100 steps; baseline '
             '4a2311d checks each generation. Presentation/control overhead changes are included. '
             'The digest includes full Python MT/Gaussian, NumPy and Torch CPU RNG states, final '
             'generation and legacy; it does not include final cell arrays. Exact per-cell results '
             'are checked separately by the engine regression tests. This is not a pure-kernel '
             'speedup, full-world total, concurrent-world feature or local-M1 measurement.',
    'cases': [row],
}
(args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n',
                                            encoding='utf-8')
print(json.dumps(row, allow_nan=False), flush=True)
if not row['same_legacy'] or not row['same_checked_state_and_rng_digest']:
    raise SystemExit('Inorganic legacy/state/RNG differs: reject the speedup until investigated')
