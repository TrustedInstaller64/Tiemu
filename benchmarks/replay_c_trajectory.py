"""Hosted-runner C experiment: frozen source, bounded human phases, raw snapshots.

No world getter is called by the observer. Timing with --observe is diagnostic
only: separate plain processes supply performance evidence. Snapshots identify
the first differing *checkpoint*, not necessarily the first causal interaction.
"""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import types

import numpy as np
import torch


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(clean(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rng_state():
    return {'python': digest(random.getstate()), 'numpy': digest(np.random.get_state()),
            'torch': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def networks(sim, values=False):
    result = {}
    for name, model, optimizer in (
            ('action', sim.action_policy_network, sim.action_optimizer),
            ('value', sim.value_network, sim.value_optimizer),
            ('guide', sim.guide_network, sim.guide_optimizer)):
        h = hashlib.sha256()
        flattened = [] if values else None

        def consume(label, value):
            h.update(label.encode())
            if torch.is_tensor(value):
                array = value.detach().cpu().numpy()
                h.update(str((str(array.dtype), array.shape)).encode())
                h.update(array.tobytes())
                if values:
                    flattened.extend(array.reshape(-1).tolist())
            else:
                h.update(json.dumps(clean(value), sort_keys=True).encode())
                if values and isinstance(value, (int, float)):
                    flattened.append(float(value))

        for key, tensor in model.state_dict().items():
            consume('parameter.'+key, tensor)
        for group_index, group in enumerate(optimizer.param_groups):
            for key, value in sorted(group.items()):
                if key != 'params':
                    consume(f'group.{group_index}.{key}', value)
            for index, parameter in enumerate(group['params']):
                for key, value in sorted(optimizer.state.get(parameter, {}).items()):
                    consume(f'adam.{group_index}.{index}.{key}', value)
        result[name] = clean(flattened) if values else h.hexdigest()
    return result


def snapshot(sim, culled_names=()):
    before_rng = rng_state()
    population = []
    path_counts = [0] * 12
    unknown_paths = 0
    def entity_state(entity):
        if entity is None:
            return None
        row = {key: clean(value) for key, value in vars(entity).items()
               if key != '_titan_to_path_model'}
        row['cached_paths'] = row.pop('_path_affinities')
        return row
    # The detached distribution calculation reads the cache when present; when
    # absent it computes a reporting projection without filling the world cache.
    for entity in sim.population:
        raw = vars(entity)
        population.append(entity_state(entity))
        paths = raw['_path_affinities']
        if paths is None:
            paths = np.maximum(0, np.dot(raw['titan_affinities'],
                                        raw['_titan_to_path_model'].titan_to_path_matrix))
        if np.all(np.isfinite(paths)):
            dominant = int(np.argmax(paths))
            path_counts[dominant] += 1
            population[-1]['reporting_dominant_idx'] = dominant
        else:
            unknown_paths += 1
            population[-1]['reporting_dominant_idx'] = None
    manager = sim.population_manager
    result = {'population': population,
              'reincarnator': entity_state(sim.reincarnator),
              'reincarnator_population_index': next(
                  (index for index, entity in enumerate(sim.population) if entity is sim.reincarnator), None),
              'dominant_path_counts': path_counts, 'unknown_path_count': unknown_paths,
              'seats': clean(sim.parliament_manager.seats),
              'culled_names': list(culled_names),
              'blueprint': clean(sim.base_titan_affinities),
              'zeitgeist': clean(sim.cosmic_zeitgeist),
              'rng': before_rng, 'networks': networks(sim),
              'managers': {'mutation_rate': clean(manager.mutation_rate),
                           'max_affinity_norm': clean(manager.max_affinity_norm),
                           'last_diversity': clean(sim.last_diversity),
                           'highest_avg_score': clean(sim.highest_avg_score),
                           'last_baie_score': clean(getattr(sim, 'last_baie_score', None))}}
    if rng_state() != before_rng:
        raise AssertionError('Snapshot observer consumed random numbers')
    return result


class QuietDebugger:
    def __init__(self, *args, **kwargs):
        self.paused, self.stop_requested, self.last_command = False, False, ''
    def poll(self): pass
    def close(self): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('original', 'a', 'c'), required=True)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--population', type=int, default=1000)
    parser.add_argument('--generations', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--reincarnator', action='store_true')
    parser.add_argument('--native-threads', type=int, default=1)
    parser.add_argument('--native-min-rows', type=int, default=4096)
    args = parser.parse_args()
    if min(args.population, args.generations, args.native_threads, args.native_min_rows) < 1:
        parser.error('counts must be positive')
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from config import config
    config['llm']['enable_llm'] = False
    config['llm']['kaoselanna_llm_enabled'] = False
    config['log']['enable'] = False
    config.setdefault('runtime', {}).update(
        torch_device='cpu', interactive=False, cpu_threads=1,
        native_threads=args.native_threads, native_min_rows=args.native_min_rows,
        experimental_c_scoring=args.mode == 'c')
    experimental_calls = {'count': 0}
    if args.mode != 'original':
        from native_backend import configure_backend
        native = configure_backend('cpp')
        if Path(native.__core_file__).resolve().parent != root:
            raise AssertionError('Native core does not belong to requested source root')
        scorer = getattr(native.BatchEvaluator, 'concepts_zeitgeist', None)
        if scorer is not None:
            def counted_scoring(self, *positional, **keywords):
                experimental_calls['count'] += 1
                return scorer(self, *positional, **keywords)
            native.BatchEvaluator.concepts_zeitgeist = counted_scoring
        elif args.mode == 'c':
            raise AssertionError('Experimental C scoring method is absent')
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    console = types.ModuleType('debugger')
    console.Debugger = QuietDebugger
    sys.modules['debugger'] = console
    import simulation
    AeonEvolution = simulation.AeonEvolution
    entry_counts = {'calls': 0, 'successes': 0}
    if args.mode != 'original':
        original_entry = simulation.try_run_encounters

        def counted_entry(*positional, **keywords):
            entry_counts['calls'] += 1
            result = original_entry(*positional, **keywords)
            entry_counts['successes'] += result is not None
            return result
        simulation.try_run_encounters = counted_entry
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    options = {key: value for key, value in config['simulation'].items()
               if key != 'elites_to_keep_in_cycle'}
    options.update(num_initial_entities=args.population, population_soft_cap=args.population,
                   population_hard_cap=2*args.population, bard_frequency=0,
                   laertes_frequency=0, kaoselanna_llm_enabled=False)
    sim = AeonEvolution(**options)
    try:
        sim._create_initial_population(create_reincarnator=args.reincarnator)
        sim.legacy_manager.newborn_affinity_modifier = np.zeros(12)
        sim.generation, sim.total_generations = 176200, 335503360
        snapshots = []
        if args.observe:
            snapshots.append({'step': 0, 'stage': 'initial', 'state': snapshot(sim)})
        wall_seconds = cpu_seconds = 0.0
        culled_names = []
        for step in range(1, args.generations+1):
            cpu_start, wall_start = time.process_time(), time.perf_counter()
            culled = sim._run_one_generation()
            wall_seconds += time.perf_counter()-wall_start
            cpu_seconds += time.process_time()-cpu_start
            culled_names = sorted(entity.name for entity in culled)
            if args.observe:
                snapshots.append({'step': step, 'stage': 'after_interactions',
                                  'state': snapshot(sim, culled_names)})
            cpu_start, wall_start = time.process_time(), time.perf_counter()
            sim._evolve_and_grow(culled)
            sim.generation += 1
            wall_seconds += time.perf_counter()-wall_start
            cpu_seconds += time.process_time()-cpu_start
            if args.observe:
                snapshots.append({'step': step, 'stage': 'after_growth',
                                  'state': snapshot(sim, culled_names)})
        final_state = snapshot(sim, culled_names)
        final_network_values = networks(sim, values=True)
        if args.mode != 'original' and entry_counts['successes'] != args.generations:
            raise AssertionError(f'Native encounter fallback invalidated measurement: {entry_counts}')
        if args.mode == 'c' and not experimental_calls['count']:
            raise AssertionError('Experimental C scoring was not executed')
        result = {
            'schema_version': 1,
            'meta': {'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                                       text=True, encoding='utf-8').strip(),
                     'mode': args.mode, 'seed': args.seed, 'population': args.population,
                     'generations': args.generations, 'reincarnator': args.reincarnator,
                     'observed': args.observe, 'native_threads': args.native_threads,
                     'platform': platform.platform(), 'machine': platform.machine(),
                     'python': platform.python_version(), 'numpy': np.__version__,
                     'torch': torch.__version__, 'native_encounters': entry_counts,
                     'experimental_calls': experimental_calls['count']},
            'timing': {'wall_seconds': wall_seconds, 'cpu_seconds': cpu_seconds,
                       'usable_for_performance': not args.observe},
            'snapshots': snapshots, 'final_state': final_state,
            'final_state_digest': digest(final_state), 'final_network_values': final_network_values,
            'scope': 'Bounded human phase methods, same seed/dependencies/device, no LLM/display/export. '
                     'Initialization, snapshots, cull-name reporting and process startup excluded from timing. '
                     'Observed timing is never performance evidence. Checkpoints bracket generation and '
                     'growth, not individual encounters. Reporting paths use raw cache if present, detached '
                     'NumPy projection otherwise; no lazy getters or world writes. Names are reusable after '
                     'death, so individual comparisons cease after discrete divergence. No full-world claim.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=True, allow_nan=False,
                                          separators=(',', ':'))+'\n', encoding='utf-8')
        print(json.dumps({'mode': args.mode, 'seed': args.seed, 'observed': args.observe,
                          'wall_seconds': wall_seconds, 'final_state_digest': result['final_state_digest']}))
    finally:
        sim.close() if hasattr(sim, 'close') else sim.debugger.close()


if __name__ == '__main__':
    main()
