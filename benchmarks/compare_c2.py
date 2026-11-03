"""Bounded hosted C2 validation; original is restricted to 1,000-entity worlds.

Each subprocess runs one world. This is validation infrastructure, not a product
multi-world scheduler. Small-original, large-native and births are separate jobs.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from c_divergence import compare_runs


def configurations(original, a2, c1, candidate):
    def entry(name, root, mode, newborns=False, workers=1, threshold=4096):
        return {'name': name, 'root': str(Path(root).resolve()), 'mode': mode,
                'newborns': newborns, 'workers': workers, 'threshold': threshold}
    return [entry('original', original, 'original'), entry('a2', a2, 'a'),
            entry('c1', c1, 'c'), entry('off', candidate, 'a'),
            entry('c1only', candidate, 'c'),
            entry('c2_1', candidate, 'c', True, 1, 1),
            entry('c2_2', candidate, 'c', True, 2, 1),
            entry('c2_4', candidate, 'c', True, 4, 1)]


def selected_cases(suite):
    if suite in ('paired', 'small-original'):
        return [(1000, 6, False), (1000, 6, True)]
    if suite == 'human':
        return [(1000, 6, False)]
    if suite == 'reincarnator':
        return [(1000, 6, True)]
    if suite == 'large-native':
        return [(10000, 6, False)]
    raise ValueError('Unknown trajectory suite')


def command_for(script, configuration, output, population, generations, seed, reincarnator, observe):
    command = [sys.executable, '-B', str(script), '--root', configuration['root'],
               '--mode', configuration['mode'], '--population', str(population),
               '--generations', str(generations), '--seed', str(seed), '--output', str(output),
               '--native-threads', str(configuration['workers']),
               '--native-min-rows', str(configuration['threshold'])]
    if configuration['newborns']:
        command.append('--experimental-c-newborns')
    if reincarnator:
        command.append('--reincarnator')
    if observe:
        command.append('--observe')
    return command


def identity_gates(runs):
    return {'off_matches_a2': runs['off']['final_state_digest'] == runs['a2']['final_state_digest'],
            'c1only_matches_frozen_c1': runs['c1only']['final_state_digest'] == runs['c1']['final_state_digest'],
            'c2_thread_digests_equal': len({runs[name]['final_state_digest']
                                           for name in ('c2_1', 'c2_2', 'c2_4')}) == 1}


def speed_report(samples):
    medians = {name: statistics.median(row['wall_seconds'] for row in rows)
               for name, rows in samples.items()}
    return {'samples': samples, 'median_seconds': medians,
            'speedup_over_original': ({name: medians['original']/value for name, value in medians.items()}
                                      if 'original' in medians else None),
            'speedup_over_a2': {name: medians['a2']/value for name, value in medians.items()},
            'speedup_over_c1': {name: medians['c1']/value for name, value in medians.items()},
            'speedup_over_c1only': {name: medians['c1only']/value for name, value in medians.items()},
            'speedup_over_c2_one_thread': {name: medians['c2_1']/medians[name]
                                          for name in ('c2_1', 'c2_2', 'c2_4')}}


def compare_births(configs, output, env):
    script = Path(__file__).with_name('benchmark_c2_births.py').resolve()
    cases = []
    for birth_count in (1000, 10000):
        samples = {cfg['name']: [] for cfg in configs}
        kernel_samples = {cfg['name']: [] for cfg in configs}
        digests = {cfg['name']: set() for cfg in configs}
        kernel_digests = {cfg['name']: set() for cfg in configs}
        comparisons, contracts, metadata = {}, {}, {}
        thread_digest_matches = []
        for repeat in range(3):
            runs = {}
            for cfg in configs[repeat:]+configs[:repeat]:
                name = cfg['name']
                destination = output/f'births-{birth_count}-{name}-{repeat}.json'
                command = [sys.executable, '-B', str(script), '--root', cfg['root'],
                           '--mode', cfg['mode'], '--existing', '1000', '--births', str(birth_count),
                           '--seed', '731', '--native-threads', str(cfg['workers']),
                           '--native-min-rows', str(cfg['threshold']), '--output', str(destination)]
                if cfg['newborns']:
                    command.append('--experimental-c-newborns')
                subprocess.run(command, check=True, cwd=output, env=env, timeout=180,
                               stdout=subprocess.DEVNULL)
                run = json.loads(destination.read_text(encoding='utf-8'))
                samples[name].append(run['timing'])
                digests[name].add(run['final_state_digest'])
                if run['kernel'] is not None:
                    kernel_samples[name].append(run['kernel'])
                    kernel_digests[name].add(run['kernel']['output_digest'])
                runs[name] = {'path': str(destination), 'final_state_digest': run['final_state_digest']}
                if repeat == 0:
                    contracts[name], metadata[name] = run['birth_contract'], run['meta']
                del run
            gates = identity_gates(runs)
            if not gates['off_matches_a2'] or not gates['c1only_matches_frozen_c1']:
                raise AssertionError(f'Disabled-feature birth regression: {gates}')
            thread_digest_matches.append(gates['c2_thread_digests_equal'])
            if repeat == 0:
                pairs = [('original', name) for name in ('a2', 'c1', 'c2_1', 'c2_2', 'c2_4')]
                pairs += [('a2', 'c2_1'), ('c1', 'c2_1'), ('c2_1', 'c2_2'), ('c2_1', 'c2_4')]
                for reference, candidate in pairs:
                    left = json.loads(Path(runs[reference]['path']).read_text(encoding='utf-8'))
                    right = json.loads(Path(runs[candidate]['path']).read_text(encoding='utf-8'))
                    comparisons[f'{candidate}_vs_{reference}'] = compare_runs(left, right)
                    del left, right
        compact_contracts = {name: {key: value for key, value in contract.items() if key != 'names'}
                             for name, contract in contracts.items()}
        report = {'case': f'births-existing1000-new{birth_count}', 'seed': 731,
                  'full_entry': speed_report(samples),
                  'kernel_samples': {name: rows for name, rows in kernel_samples.items() if rows},
                  'kernel_median_seconds': {name: statistics.median(row['wall_seconds'] for row in rows)
                                            for name, rows in kernel_samples.items() if rows},
                  'repeat_state_consistent': {name: len(values) == 1 for name, values in digests.items()},
                  'kernel_repeat_consistent': {name: len(values) == 1 for name, values in kernel_digests.items()
                                               if values},
                  'c2_thread_digests_equal_each_repeat': thread_digest_matches,
                  'birth_contracts': compact_contracts, 'execution_metadata': metadata,
                  'comparisons': comparisons, 'off_matches_a2': True,
                  'c1only_matches_frozen_c1': True,
                  'scope': 'Full single-call newborn entry reuses the pool warmed during initial '
                           'creation when C2 is active; separate prepacked native boundary uses a fresh '
                           'cold pool. Three rotated independent processes, seed731. '
                           'No measured kernel exists for original/A2/frozenC1. A kernel-only gain cannot '
                           'be used as a full-world multiplier. Fixed nonzero legacy/tide; no training.'}
        compact = {name: {key: value for key, value in comparison.items() if key != 'checkpoints'}
                   for name, comparison in comparisons.items()}
        print('C2_BIRTHS '+json.dumps({**report, 'comparisons': compact}, allow_nan=False), flush=True)
        cases.append(report)
        (output/'births-summary.json').write_text(json.dumps({'cases': cases}, allow_nan=False)+'\n',
                                                 encoding='utf-8')
        if not all(report['repeat_state_consistent'].values()) or not all(report['kernel_repeat_consistent'].values()):
            raise AssertionError('Repeated birth sample changed state; inspect saved evidence')


def main():
    parser = argparse.ArgumentParser()
    for name in ('original', 'a2', 'c1', 'candidate', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--suite', '--case', dest='suite',
                        choices=('paired', 'small-original', 'human', 'reincarnator', 'large-native', 'births'),
                        required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations(args.original, args.a2, args.c1, args.candidate)
    if args.suite == 'large-native':
        configs = [cfg for cfg in configs if cfg['name'] != 'original']
    script = Path(__file__).with_name('replay_c_trajectory.py').resolve()
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1'}
    if args.suite == 'births':
        compare_births(configs, output, env)
        return

    def execute(cfg, population, generations, seed, reincarnator, label, observe=False):
        destination = output/f'{label}-{cfg["name"]}.json'
        subprocess.run(command_for(script, cfg, destination, population, generations,
                                   seed, reincarnator, observe), check=True, cwd=output,
                       env=env, timeout=240, stdout=subprocess.DEVNULL)
        run = json.loads(destination.read_text(encoding='utf-8'))
        # Keep only small metadata; reload two trajectory files at a time below.
        return {'path': str(destination), 'meta': run['meta'], 'timing': run['timing'],
                'final_state_digest': run['final_state_digest']}

    cases = []
    for population, generations, reincarnator in selected_cases(args.suite):
        tag = f'human-{population}-{generations}-reinc{int(reincarnator)}'
        samples = {cfg['name']: [] for cfg in configs}
        sample_digests = {name: set() for name in samples}
        seed_results = []
        for seed in ((731,) if args.suite == 'large-native' else (731, 42, 2026)):
            observed, plain = {}, {}
            for cfg in configs:
                name = cfg['name']
                observed[name] = execute(cfg, population, generations, seed, reincarnator,
                                         f'{tag}-{seed}-observed', True)
                plain[name] = execute(cfg, population, generations, seed, reincarnator,
                                      f'{tag}-{seed}-plain')
                if observed[name]['final_state_digest'] != plain[name]['final_state_digest']:
                    raise AssertionError(f'Observer changed {tag}/{seed}/{name}')
                if seed == 731:
                    samples[name].append(plain[name]['timing'])
                    sample_digests[name].add(plain[name]['final_state_digest'])
            gates = identity_gates(plain)
            if not gates['off_matches_a2'] or not gates['c1only_matches_frozen_c1']:
                raise AssertionError(f'Disabled-feature regression: {gates}')
            # Thread differences remain explicit research evidence, not an
            # exception that discards the trajectory needed to investigate them.
            pairs = [('a2', 'c2_1'), ('c1', 'c2_1'), ('c2_1', 'c2_2'), ('c2_1', 'c2_4')]
            if 'original' in plain:
                pairs = [('original', name) for name in ('a2', 'c1', 'c2_1', 'c2_2', 'c2_4')]+pairs
            comparisons = {}
            for reference, candidate in pairs:
                left = json.loads(Path(observed[reference]['path']).read_text(encoding='utf-8'))
                right = json.loads(Path(observed[candidate]['path']).read_text(encoding='utf-8'))
                comparisons[f'{candidate}_vs_{reference}'] = compare_runs(left, right)
                del left, right
            row = {'case': tag, 'seed': seed, 'observer_unchanged': True, **gates,
                   'execution_metadata': {name: run['meta'] for name, run in plain.items()},
                   'comparisons': comparisons}
            seed_results.append(row)
            compact = {key: {field: value for field, value in record.items() if field != 'checkpoints'}
                       for key, record in comparisons.items()}
            print('C2_DEVIATION '+json.dumps({**row, 'comparisons': compact}, allow_nan=False), flush=True)
            # Save incrementally before starting another seed or timing process.
            (output/f'{tag}-deviation.json').write_text(json.dumps(seed_results, allow_nan=False)+'\n',
                                                        encoding='utf-8')
        for repeat in (1, 2):
            plain = {}
            for cfg in configs[repeat:]+configs[:repeat]:
                name = cfg['name']
                plain[name] = execute(cfg, population, generations, 731, reincarnator,
                                      f'{tag}-731-timing-{repeat}')
                samples[name].append(plain[name]['timing'])
                sample_digests[name].add(plain[name]['final_state_digest'])
            gates = identity_gates(plain)
            if not gates['off_matches_a2'] or not gates['c1only_matches_frozen_c1']:
                raise AssertionError(f'Disabled-feature timing regression: {gates}')
        deterministic = {name: len(digests) == 1 for name, digests in sample_digests.items()}
        timing = {'case': tag, 'seed': 731, **speed_report(samples),
                  'repeat_state_consistent': deterministic,
                  'scope': 'Three rotated plain-process timings. Original comparison exists only for '
                           'the small suite. C2 2/4 workers use forced threshold 1; shared scoring/birth '
                           'thread budgets do not isolate the birth contribution or prove core occupancy.'}
        print('C2_TIMING '+json.dumps(timing, allow_nan=False), flush=True)
        cases.append({'case': tag, 'timing': timing, 'seeds': seed_results})
        (output/'summary.json').write_text(json.dumps({'suite': args.suite,
            'platform': platform.platform(), 'cases': cases,
            'scope': 'Bounded checkpoints and independent timings. Names cease to identify comparable '
                     'individuals after discrete/RNG divergence. Distribution and timing after a fork '
                     'describe possibly different realized workloads; no long-run equivalence claim.'},
            allow_nan=False)+'\n', encoding='utf-8')
        if not all(deterministic.values()):
            raise AssertionError('Repeated fixed-configuration states differ; inspect saved evidence')


if __name__ == '__main__':
    main()
