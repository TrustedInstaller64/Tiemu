"""One isolated birth sample, executed only on a hosted runner.

The full entry includes naming, RNG, Python objects, preparation, numerical work,
and insertion. The second measurement uses prepacked inputs and a fresh native
pool; it includes the adapter/output allocation and is not bare C++ CPU timing.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import types

from replay_c_trajectory import QuietDebugger, clean, cpu_metadata, digest, rng_state, snapshot
import numpy as np
import torch


def birth_contract(population, newborns, manager):
    spans = sorted((p.titan_affinities.ctypes.data, p.titan_affinities.ctypes.data+p.titan_affinities.nbytes)
                   for p in newborns)
    return {'count': len(newborns), 'names': [p.name for p in newborns],
            'integer_zero_scores': all(type(p.score) is int and p.score == 0 for p in newborns),
            'empty_path_caches': all(p._path_affinities is None for p in newborns),
            'owned_arrays': all(p.titan_affinities.flags.owndata for p in newborns),
            'disjoint_arrays': all(left[1] <= right[0] for left, right in zip(spans, spans[1:])),
            'map_matches': all(manager.name_to_entity_map.get(p.name) is p for p in population),
            'registry_matches_population': manager.existing_names == {p.name for p in population},
            'names_unique': len({p.name for p in population}) == len(population)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('original', 'a', 'c'), required=True)
    parser.add_argument('--experimental-c-newborns', action='store_true')
    parser.add_argument('--existing', type=int, default=1000)
    parser.add_argument('--births', type=int, required=True)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--native-threads', type=int, default=1)
    parser.add_argument('--native-min-rows', type=int, default=4096)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.existing, args.births, args.native_threads, args.native_min_rows) < 1:
        parser.error('counts must be positive')
    if args.experimental_c_newborns and args.mode == 'original':
        parser.error('Original code cannot enable C2')
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from config import config
    config['llm']['enable_llm'] = config['llm']['kaoselanna_llm_enabled'] = False
    config['log']['enable'] = False
    config.setdefault('runtime', {}).update(torch_device='cpu', interactive=False, cpu_threads=1,
        native_threads=args.native_threads, native_min_rows=args.native_min_rows,
        experimental_c_scoring=args.mode == 'c', experimental_c_newborns=args.experimental_c_newborns)
    native = None
    counts = {'calls': 0, 'rows': 0, 'parallel_eligible_batches': 0}
    if args.mode != 'original':
        from native_backend import configure_backend
        native = configure_backend('cpp')
        if Path(native.__core_file__).resolve().parent != root:
            raise AssertionError('Wrong numerical core source root')
        method = getattr(native.BatchEvaluator, 'newborn_affinities', None)
        if method is not None:
            def counted(self, blueprint, noise, *positional, **keywords):
                rows = len(noise)
                counts['calls'] += 1
                counts['rows'] += rows
                counts['parallel_eligible_batches'] += (
                    args.native_threads > 1 and rows > 1 and rows >= args.native_min_rows)
                return method(self, blueprint, noise, *positional, **keywords)
            native.BatchEvaluator.newborn_affinities = counted
        elif args.experimental_c_newborns:
            raise AssertionError('C2 native method missing')
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    console = types.ModuleType('debugger')
    console.Debugger = QuietDebugger
    sys.modules['debugger'] = console
    from simulation import AeonEvolution
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    options = {k: v for k, v in config['simulation'].items() if k != 'elites_to_keep_in_cycle'}
    options.update(num_initial_entities=args.existing, population_soft_cap=args.existing,
                   population_hard_cap=args.existing+args.births, bard_frequency=0,
                   laertes_frequency=0, kaoselanna_llm_enabled=False)
    sim = AeonEvolution(**options)
    try:
        sim._create_initial_population(create_reincarnator=False)
        if len(sim.population) != args.existing:
            raise AssertionError('Isolated birth fixture did not create the requested existing population')
        initial_newborn_calls = counts.copy()
        manager = sim.population_manager
        # Fixed nonzero, mixed-sign legacy/tide exercises the entire pipeline.
        legacy = np.linspace(-0.003, 0.005, 12)
        manager.cosmic_tide_vector = np.linspace(-0.002, 0.004, 12)
        blueprint = sim.base_titan_affinities.copy()
        initial = snapshot(sim)
        count_before = counts.copy()
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        manager._add_new_entities(sim.population, args.births, blueprint,
                                  sim.cosmic_zeitgeist, legacy)
        wall_seconds = time.perf_counter()-wall_start
        cpu_seconds = time.process_time()-cpu_start
        full_counts = {key: counts[key]-count_before[key] for key in counts}
        newborns = sim.population[args.existing:]
        contract = birth_contract(sim.population, newborns, manager)
        if len(newborns) != args.births or not all(value for key, value in contract.items()
                                                  if key not in ('count', 'names')):
            raise AssertionError(f'Newborn structural contract failed: {contract}')
        if args.experimental_c_newborns and (not full_counts['calls'] or full_counts['rows'] != args.births):
            raise AssertionError(f'Birth fast path did not execute for every newborn: {full_counts}')
        final = snapshot(sim)
        kernel = None
        if native is not None and hasattr(native.BatchEvaluator, 'newborn_affinities'):
            # Independent local generator cannot consume the world RNG stream.
            noise = np.random.RandomState(args.seed+1).normal(
                0, manager.mutation_rate, (args.births, len(blueprint)))
            legacy_titans = np.dot(legacy, manager.titan_to_path_model.path_to_titan_feedback_matrix)
            evaluator = native.BatchEvaluator(args.native_threads, args.native_min_rows)
            try:
                before_rng = rng_state()
                cpu_start, wall_start = time.process_time(), time.perf_counter()
                values = evaluator.newborn_affinities(blueprint, noise, legacy_titans,
                    manager.purity_factor, manager.cosmic_tide_vector, manager.max_affinity_norm)
                kernel = {'wall_seconds': time.perf_counter()-wall_start,
                          'cpu_seconds': time.process_time()-cpu_start,
                          'rows': args.births, 'output_digest': digest(values),
                          'finite': bool(np.isfinite(values).all()),
                          'scope': 'Prepacked adapter/native boundary with fresh cold pool. Input '
                                   'preparation and handle construction excluded; output allocation, '
                                   'FFI, first worker startup and completion included. Not bare C++ '
                                   'instruction timing and not a full-birth speed claim.'}
                if before_rng != rng_state():
                    raise AssertionError('Numerical birth kernel consumed global RNG')
            finally:
                evaluator.close()
        meta = {'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                                  text=True, encoding='utf-8').strip(),
                'mode': args.mode, 'seed': args.seed, 'population': args.existing,
                'generations': 1, 'reincarnator': False, 'observed': True,
                'births': args.births, 'native_threads': args.native_threads,
                'native_min_rows': args.native_min_rows,
                'experimental_c_newborns': args.experimental_c_newborns,
                'full_entry_newborn_calls': full_counts,
                'initial_newborn_calls': initial_newborn_calls,
                'platform': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
                'cpu': cpu_metadata()}
        result = {'schema_version': 1, 'meta': meta,
                  'timing': {'wall_seconds': wall_seconds, 'cpu_seconds': cpu_seconds,
                             'usable_for_performance': True},
                  'kernel': kernel, 'birth_contract': contract,
                  'snapshots': [{'step': 0, 'stage': 'initial', 'state': initial},
                                {'step': 1, 'stage': 'after_growth', 'state': final}],
                  'final_state': final, 'final_state_digest': digest(final),
                  'final_network_values': {},
                  'scope': 'One isolated _add_new_entities call including names, RNG, Python objects, '
                           'distribution preparation, native invocation and writeback. Existing world '
                           'initialization and bracketing snapshots excluded. Full entry reuses the '
                           'pool warmed during initial creation when C2 is active; the separate '
                           'prepacked call uses a fresh cold pool. This is not an observed/plain '
                           'trajectory gate. Nonzero fixed legacy/tide. '
                           'Full-entry and prepacked-kernel timings are distinct; kernels do not '
                           'establish equivalent full-program gains. Snapshots only bracket the call.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clean(result), ensure_ascii=True, allow_nan=False,
                                          separators=(',', ':'))+'\n', encoding='utf-8')
        print(json.dumps({'births': args.births, 'seconds': wall_seconds,
                          'newborn_calls': full_counts, 'final_state_digest': result['final_state_digest']}))
    finally:
        sim.close() if hasattr(sim, 'close') else sim.debugger.close()


if __name__ == '__main__':
    main()
