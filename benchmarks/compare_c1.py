"""Bounded remote C1 validation, not a multi-world product scheduler."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from c_divergence import compare_runs


def main():
    parser = argparse.ArgumentParser()
    for name in ('original', 'a2', 'candidate', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', choices=('all', 'human', 'large', 'reincarnator', 'large-native'),
                        default=os.environ.get('C1_CASE', 'all'))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configurations = [('original', args.original.resolve(), 'original'),
                      ('a2', args.a2.resolve(), 'a'),
                      ('off', args.candidate.resolve(), 'a'),
                      ('c1', args.candidate.resolve(), 'c')]
    script = Path(__file__).with_name('replay_c_trajectory.py').resolve()
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8',
           'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}

    def execute(configuration, population, generations, reincarnator, seed, label, observe=False):
        name, root, mode = configuration
        destination = output / f'{label}-{name}.json'
        command = [sys.executable, '-B', str(script), '--root', str(root), '--mode', mode,
                   '--population', str(population), '--generations', str(generations),
                   '--seed', str(seed), '--output', str(destination)]
        if observe:
            command.append('--observe')
        if reincarnator:
            command.append('--reincarnator')
        subprocess.run(command, check=True, env=env, cwd=output, timeout=600,
                       stdout=subprocess.DEVNULL)
        return json.loads(destination.read_text(encoding='utf-8'))

    if args.case == 'large-native':
        # Original 10k trajectories were already compared in the main run.
        # This bounded continuation measures only the incremental C1 gain;
        # it must never manufacture an original-Python speed ratio.
        native_configurations = configurations[1:]
        samples = {name: [] for name, _, _ in native_configurations}
        digests = {name: set() for name in samples}
        for repeat in range(3):
            runs = {}
            for configuration in native_configurations[repeat:] + native_configurations[:repeat]:
                name = configuration[0]
                run = execute(configuration, 10000, 4, False, 731,
                              f'large-native-731-timing-{repeat}')
                runs[name] = run
                samples[name].append(run['timing'])
                digests[name].add(run['final_state_digest'])
            if runs['a2']['final_state_digest'] != runs['off']['final_state_digest']:
                raise AssertionError('Disabled C experiment changed frozen A2 state')
        if any(len(values) != 1 for values in digests.values()):
            raise AssertionError('Repeated native timing samples are not deterministic')
        medians = {name: statistics.median(x['wall_seconds'] for x in rows)
                   for name, rows in samples.items()}
        report = {'case': 'human-10000-4-reinc0', 'seed': 731, 'samples': samples,
                  'median_seconds': medians, 'off_matches_a2': True,
                  'repeat_state_consistent': True, 'c1_speedup_over_original': None,
                  'c1_speedup_over_a2': medians['a2'] / medians['c1'],
                  'c1_speedup_over_off': medians['off'] / medians['c1'],
                  'scope': 'Native-only timing continuation; three rotated seed731 repeats. '
                           'No original-Python timing or additional trajectory claim.'}
        (output / 'native-timing-summary.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('C1_TIMING ' + json.dumps(report, allow_nan=False), flush=True)
        return

    all_cases = []
    cases = {'human': (1000, 12, False), 'large': (10000, 4, False),
             'reincarnator': (1000, 12, True)}
    selected = list(cases.values()) if args.case == 'all' else [cases[args.case]]
    for population, generations, reincarnator in selected:
        tag = f'human-{population}-{generations}-reinc{int(reincarnator)}'
        samples = {name: [] for name, _, _ in configurations}
        seed_rows = []
        for seed in (731, 42, 2026):
            observed, plain = {}, {}
            for configuration in configurations:
                name = configuration[0]
                observed[name] = execute(configuration, population, generations, reincarnator,
                                         seed, f'{tag}-{seed}-observed', observe=True)
                plain[name] = execute(configuration, population, generations, reincarnator,
                                      seed, f'{tag}-{seed}-plain')
                if observed[name]['final_state_digest'] != plain[name]['final_state_digest']:
                    raise AssertionError(f'Observer changed {tag}, seed={seed}, mode={name}')
                if seed == 731:
                    samples[name].append(plain[name]['timing'])
            if plain['a2']['final_state_digest'] != plain['off']['final_state_digest']:
                raise AssertionError('Disabled C experiment changed frozen A2 state')
            pairs = {}
            for reference, candidate in [('original', 'a2'), ('original', 'c1'), ('a2', 'c1')]:
                comparison = compare_runs(observed[reference], observed[candidate])
                pairs[f'{candidate}_vs_{reference}'] = comparison
            row = {'seed': seed, 'observer_unchanged': True, 'off_matches_a2': True,
                   'comparisons': pairs}
            seed_rows.append(row)
            # Compact per-seed evidence survives artifact expiry in job logs.
            compact = {name: {key: value for key, value in comparison.items()
                              if key != 'checkpoints'} for name, comparison in pairs.items()}
            print('C1_DEVIATION ' + json.dumps({'case': tag, **row, 'comparisons': compact},
                                             allow_nan=False), flush=True)
        # The first timing sample above is reused; rotate the remaining fresh
        # processes. Never mix different seeds into a speed median.
        for repeat in (1, 2):
            for configuration in configurations[repeat:] + configurations[:repeat]:
                run = execute(configuration, population, generations, reincarnator, 731,
                              f'{tag}-731-timing-{repeat}')
                samples[configuration[0]].append(run['timing'])
        medians = {name: statistics.median(x['wall_seconds'] for x in rows)
                   for name, rows in samples.items()}
        timing = {'case': tag, 'seed': 731, 'samples': samples, 'median_seconds': medians,
                  'c1_speedup_over_original': medians['original'] / medians['c1'],
                  'c1_speedup_over_a2': medians['a2'] / medians['c1'],
                  'c1_speedup_over_off': medians['off'] / medians['c1']}
        print('C1_TIMING ' + json.dumps(timing), flush=True)
        case = {'case': tag, 'timing': timing, 'seeds': seed_rows}
        all_cases.append(case)
        (output / 'summary.json').write_text(json.dumps({
            'platform': platform.platform(), 'cases': all_cases,
            'scope': 'Same runner; CPU; fixed seeds; finite human-phase checkpoints. '
                     'Three unobserved timing samples at seed 731, rotated process order. '
                     'Initialization/model startup and final capture excluded; first native pool '
                     'start and phase preparation/writeback included. No LLM/output/export. '
                     'Observed timing is excluded. First divergence is checkpoint resolution, '
                     'not the exact first encounter. Not long-run statistical equivalence.'},
            indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
