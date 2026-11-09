"""Frozen A2/A3 diagnosis of the short macOS aeonic result.

This reuses the existing fresh-world fixture and raw state/RNG/Adam checks.
Independent, counterbalanced process blocks are the timing units. The duplicate
A3 configuration measures same-code variability; observers never supply speedup
estimates. No production GC setting, optimizer order or simulation rule changes.
"""
import argparse
import functools
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

# This module imports only the standard library at top level. Its project,
# NumPy and Torch imports occur inside worker(), after selecting source-root.
import diagnose_aeonic as fixture


EXPECTED_UPDATES = 15
ZERO_COUNTS = (
    'native.BatchEvaluator.concepts', 'ResidentWorld.__init__',
    'ResidentWorld.births', 'ResidentWorld.reconcile',
    'world.sources', 'world.pack_sources',
)


class Timers(fixture.Timers):
    """Observe the frozen paths without modifying protected eligibility methods."""

    def install(self, sim, native, torch):
        import numpy as np
        from aeonic_cycle_manager import AeonicCycleManager

        self.training_depth = 0
        self.gc_started = {}
        self.gc_seconds = [0.0, 0.0, 0.0]
        self.gc_events = [0, 0, 0]
        self.gc_unclosed = False

        def observe_gc(phase, info):
            generation = info['generation']
            if phase == 'start':
                self.gc_started[generation] = time.perf_counter()
            elif phase == 'stop':
                start = self.gc_started.pop(generation, None)
                if start is not None:
                    self.gc_seconds[generation] += time.perf_counter() - start
                    self.gc_events[generation] += 1

        self.gc_callback = observe_gc
        gc.callbacks.append(observe_gc)
        for name in ('run_aeonic_cycle_generation', 'select_aeonic_opponents', 'end_aeonic_cycle'):
            self.replace(AeonicCycleManager, name, self.wrap(
                f'AeonicCycleManager.{name}', getattr(AeonicCycleManager, name)))
        original = AeonicCycleManager.aeonic_interaction
        training = self.wrap('interaction.reincarnator', original)
        ordinary = self.wrap('interaction.other', original)

        @functools.wraps(original)
        def interaction(manager, attacker, defender, reincarnator, *positional, **keywords):
            is_training = attacker is reincarnator
            self.training_depth += int(is_training)
            try:
                return (training if is_training else ordinary)(
                    manager, attacker, defender, reincarnator, *positional, **keywords)
            finally:
                self.training_depth -= int(is_training)

        self.replace(AeonicCycleManager, 'aeonic_interaction', interaction)
        for label, model in (('actor.forward', sim.action_policy_network),
                             ('value.forward', sim.value_network)):
            self.replace(model, 'forward', self.wrap(label, model.forward))
        for label, optimizer in (('actor', sim.action_optimizer), ('value', sim.value_optimizer)):
            for name in ('zero_grad', 'step'):
                self.replace(optimizer, name, self.wrap(f'{label}.{name}', getattr(optimizer, name)))
        self.replace(torch.Tensor, 'backward', self.wrap('autograd.backward', torch.Tensor.backward))

        for owner, name in ((np, 'concatenate'), (torch, 'from_numpy'), (torch, 'tensor')):
            original = getattr(owner, name)
            measured = self.wrap(f'training.input.{name}', original)

            def when_training(*positional, _original=original, _measured=measured, **keywords):
                return (_measured if self.training_depth else _original)(*positional, **keywords)

            self.replace(owner, name, when_training)
        self.replace(native.BatchEvaluator, 'concepts', self.wrap(
            'native.BatchEvaluator.concepts', native.BatchEvaluator.concepts))
        for name in ('path_affinities', 'concepts', 'normalize_pipeline', 'normalize', 'purify'):
            self.replace(native, name, self.wrap(f'native.{name}', getattr(native, name)))

        root = Path(sys.modules['simulation'].__file__).resolve().parent
        if (root/'native_world.py').is_file():
            import native_world
            for name in ('__init__', 'births', 'reconcile'):
                self.replace(native_world.ResidentWorld, name, self.wrap(
                    f'ResidentWorld.{name}', getattr(native_world.ResidentWorld, name)))
            for name, label in (('_sources', 'world.sources'), ('_pack_sources', 'world.pack_sources')):
                if hasattr(native_world, name):
                    self.replace(native_world, name, self.wrap(label, getattr(native_world, name)))
        # Do not wrap PopulationManager, ParliamentManager, Pathstrider or
        # TitanToPathModel methods: A3 checks their identities before native work.

    def restore(self):
        callback = getattr(self, 'gc_callback', None)
        if callback is not None and callback in gc.callbacks:
            gc.callbacks.remove(callback)
        self.gc_unclosed = bool(getattr(self, 'gc_started', {}))
        super().restore()

    def summary(self, generations, updates):
        result = super().summary(generations, updates)
        for name in ZERO_COUNTS:
            if name not in self.rows:
                result['spans'].append({'name': name, 'calls': 0, 'inclusive_seconds': 0.0,
                                        'exclusive_instrumented_seconds': 0.0})
        result['gc_observer_seconds'] = self.gc_seconds
        result['gc_observer_events'] = self.gc_events
        if self.gc_unclosed or self.training_depth:
            result['errors'].append('unclosed GC callback or training scope')
        if self.rows.get('native.BatchEvaluator.concepts', {}).get('calls', 0):
            result['errors'].append('aeonic fixture unexpectedly used the human batch-scoring pool')
        result['valid'] = not result['errors']
        result['scope'] += (
            ' Input spans cover concatenate/from_numpy/tensor calls inside reincarnator interactions, '
            'not every tensor conversion or loss/distribution operation. Protected native eligibility '
            'methods are not wrapped. GC callback duration covers the observer install/restore window, '
            'including small pre/post-loop bookkeeping; exact loop collection counts remain in the '
            'sample gc_collections field. GC is never disabled. Zero World counts can rule out '
            'newborn/resident fixed costs in this fixture, not in every possible aeonic world.')
        return result


def hardware_snapshot():
    info = {'platform': platform.platform(), 'machine': platform.machine(),
            'logical_cpus': os.cpu_count(), 'processor': platform.processor()}
    if platform.system() == 'Darwin':
        for key in ('machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.memsize'):
            completed = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True, timeout=5)
            info[key] = completed.stdout.strip() if completed.returncode == 0 else None
    info['scope'] = 'Read-only runner metadata; not sustained frequency, core placement or power evidence.'
    return info


def worker(args):
    # Use exactly the existing setup and digest, but substitute the narrowly
    # scoped observer above. Both timing and observation preserve default GC.
    args.mode = 'plain'
    hardware = hardware_snapshot()
    started = time.time()
    fixture.Timers = Timers
    fixture.worker(args)
    path = args.output_dir/'worker.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if any(run['optimizer_updates'] != EXPECTED_UPDATES for run in record['runs']):
        raise AssertionError('the frozen aeonic fixture must perform exactly 15 updates per optimizer')
    record.update(hardware=hardware, worker_started_unix=started, worker_finished_unix=time.time(),
                  expected_optimizer_updates=EXPECTED_UPDATES)
    path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')


def spread(values):
    quartiles = statistics.quantiles(values, n=4, method='inclusive')
    return {'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values),
            'q1': quartiles[0], 'q3': quartiles[2]}


def summarize_observers(records):
    runs = [run for record in records for run in record['runs']]
    names = sorted({span['name'] for run in runs for span in run['observation']['spans']})
    spans = []
    for name in names:
        rows = [next((span for span in run['observation']['spans'] if span['name'] == name),
                     {'calls': 0, 'inclusive_seconds': 0.0, 'exclusive_instrumented_seconds': 0.0})
                for run in runs]
        spans.append({'name': name, 'call_counts': sorted({row['calls'] for row in rows}),
                      'median_inclusive_seconds': statistics.median(row['inclusive_seconds'] for row in rows),
                      'median_exclusive_instrumented_seconds': statistics.median(
                          row['exclusive_instrumented_seconds'] for row in rows)})
    return {'runs': len(runs), 'valid': all(run['observation']['valid'] for run in runs),
            'spans': sorted(spans, key=lambda row: row['median_inclusive_seconds'], reverse=True),
            'gc_collections': [run['gc_collections'] for run in runs],
            'gc_observer_seconds': [run['observation']['gc_observer_seconds'] for run in runs],
            'scope': runs[0]['observation']['scope']}


def parent(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    configurations = [('A2', baseline), ('A3', candidate), ('A3-identical-repeat', candidate)]
    report = {'baseline': str(baseline), 'candidate': str(candidate), 'cases': [],
              'scope': 'Frozen A2/A3 on the same hosted macOS runner. Every six blocks cover all '
                       'three-configuration order permutations; duplicate A3 processes quantify '
                       'same-code variability, not another optimization. Each process uses two '
                       'fresh warmup worlds, then fresh measured worlds with standard GC. Setup, '
                       'state hashing, teardown and imports are excluded from phase timings. '
                       'Observer runs are separate and never used to report speedups. CPU/Torch/'
                       'BLAS/native threads are one; real ordered Adam updates are checked as 15 '
                       'per optimizer. This is a bounded diagnostic, not product batch scheduling, '
                       'a local M1 test or a proof of arbitrary long-run equivalence.'}
    for generations in (2, 12):
        runs = {name: [] for name, _ in configurations}
        orders = []

        def execute(name, root, block, observe=False):
            destination = output/f'g{generations}-{name}-{block}'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--source-root', str(root), '--baseline', str(baseline),
                       '--output-dir', str(destination), '--generations', str(generations),
                       '--samples', '3' if observe else str(args.samples)]
            if observe:
                command.append('--observe')
            subprocess.run(command, check=True, timeout=240, stdout=subprocess.DEVNULL,
                           env={**os.environ, 'PYTHONHASHSEED': '0'})
            return json.loads((destination/'worker.json').read_text(encoding='utf-8'))

        for block in range(args.blocks):
            rotation = block % len(configurations)
            order = configurations[rotation:] + configurations[:rotation]
            if block % 2:
                order = list(reversed(order))
            orders.append([name for name, _ in order])
            for name, root in order:
                runs[name].append(execute(name, root, block))
        observations = {'A2': [], 'A3': []}
        for block in range(3):
            order = configurations[:2] if block % 2 == 0 else list(reversed(configurations[:2]))
            for name, root in order:
                observations[name].append(execute(name, root, f'observed-{block}', True))
        all_records = [record for groups in (runs, observations) for series in groups.values() for record in series]
        digests = {run['state_digest'] for record in all_records for run in record['runs']}
        if len(digests) != 1:
            raise AssertionError(f'g{generations}: pure/duplicate/observer states differ; inspect raw evidence')
        medians = {name: [record['median_seconds'] for record in series] for name, series in runs.items()}
        ratios = {name: [value/base for value, base in zip(values, medians['A2'])]
                  for name, values in medians.items() if name != 'A2'}
        duplicate_ratios = [repeat/original for repeat, original in zip(
            medians['A3-identical-repeat'], medians['A3'])]
        row = {'population': 200, 'generations': generations, 'blocks': args.blocks,
               'samples_per_process': args.samples, 'process_orders': orders,
               'source_commits': {name: records[0]['commit'] for name, records in runs.items()},
               'runtime': {name: {key: records[0][key] for key in
                                  ('python', 'numpy', 'torch', 'binding', 'core_file')}
                           for name, records in runs.items()},
               'hardware': runs['A2'][0]['hardware'],
               'process_medians_seconds': medians,
               'process_median_spread': {name: spread(values) for name, values in medians.items()},
               'paired_time_ratios_to_A2': ratios,
               'paired_time_ratio_spread': {name: spread(values) for name, values in ratios.items()},
               'geometric_mean_paired_time_ratio': {name: math.exp(statistics.mean(math.log(v) for v in values))
                                                     for name, values in ratios.items()},
               'identical_A3_repeat_over_A3': duplicate_ratios,
               'identical_A3_repeat_ratio_spread': spread(duplicate_ratios),
               'optimizer_updates_per_optimizer': EXPECTED_UPDATES,
               'pure_gc_collections': {name: [[run['gc_collections'] for run in record['runs']]
                                             for record in series] for name, series in runs.items()},
               'same_checked_state_and_rng': True,
               'observations': {name: summarize_observers(records) for name, records in observations.items()}}
        report['cases'].append(row)
        (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--generations', type=int, choices=(2, 12), default=2)
    parser.add_argument('--observe', action='store_true')
    options = parser.parse_args()
    if options.blocks < 6 or options.blocks % 6 or options.samples < 1:
        parser.error('blocks must be a positive multiple of six; samples must be positive')
    if options.worker and options.source_root is None:
        parser.error('--worker requires --source-root')
    if not options.worker and options.candidate is None:
        parser.error('the parent requires --candidate')
    worker(options) if options.worker else parent(options)
