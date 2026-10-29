"""Bounded single-world timing; intended for remote CI while the Mac is reserved.

Each process measures one source checkout and one configuration. This is not a
multi-world product runner. Setup, warm-up, final state capture and process
startup are excluded from timings; the actual phase calls remain included.
Each measured sample uses a fresh simulation: its native pool's first startup,
if the scoring threshold is reached, is included. This is not steady-state pool
throughput, and the aeonic case does not exercise the human batch-scoring pool.
"""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
import argparse
import cProfile
import functools
import hashlib
import json
import logging
from pathlib import Path
import platform
import pstats
import random
import statistics
import subprocess
import sys
import time
import types

try:
    import resource
except ImportError:
    resource = None

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--phase', choices=['human', 'aeonic'], required=True)
parser.add_argument('--population', type=int, default=1000)
parser.add_argument('--generations', type=int, default=6)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--native-threads', type=int, default=1)
parser.add_argument('--native-min-rows', type=int, default=4096)
parser.add_argument('--with-reincarnator', action='store_true',
                    help='Use the original seeded initial reincarnator selection (human profiling).')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--profile-output-dir', type=Path,
                    help='Profile only measured phase loops; write one .prof per sample (remote use).')
parser.add_argument('--boundary-timers', action='store_true',
                    help='Use explicit call wrappers instead of cProfile (remote diagnostics only).')
args = parser.parse_args()
if args.boundary_timers and args.profile_output_dir:
    parser.error('boundary timers and cProfile must run in separate samples')
if min(args.population, args.generations, args.repeats, args.native_threads, args.native_min_rows) < 1:
    parser.error('counts must be positive')
root = args.source_root.resolve()
sys.path.insert(0, str(root))
import numpy as np
import torch
from config import config
from native_backend import configure_backend

config['llm']['enable_llm'] = False
config['llm']['kaoselanna_llm_enabled'] = False
config['log']['enable'] = False
config['runtime'].update(torch_device='cpu', interactive=False,
                         native_threads=args.native_threads, native_min_rows=args.native_min_rows)
native = configure_backend('cpp')
if Path(native.__file__).resolve().parent != root:
    raise RuntimeError('Build the native extension in-place in each source root before comparing')
if Path(getattr(native, '__core_file__', native.__file__)).resolve().parent != root:
    raise RuntimeError('The shared numerical core must come from the measured source root')
logging.disable(logging.CRITICAL)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class QuietDebugger:
    def __init__(self, *a, **kw):
        self.paused = False
        self.stop_requested = False
        self.last_command = ''
    def poll(self): pass
    def close(self): pass


stub = types.ModuleType('debugger')
stub.Debugger = QuietDebugger
sys.modules['debugger'] = stub
from simulation import AeonEvolution
from constants import TITAN_NAMES


def digest_state(sim):
    digest = hashlib.sha256()
    def entity_state(entity):
        if entity is None:
            return None
        cache = entity._path_affinities
        return {**entity.to_dict(), 'activity': entity.activity, 'stability': entity.stability,
                'heroic_tendency': entity.heroic_tendency,
                'cached_path_affinities': None if cache is None else cache.tolist()}
    payload = {'population': [entity_state(p) for p in sim.population],
               'reincarnator': entity_state(sim.reincarnator),
               'reincarnator_population_index': next(
                   (i for i, p in enumerate(sim.population) if p is sim.reincarnator), None),
               'seats': sim.parliament_manager.seats,
               'blueprint': sim.base_titan_affinities.tolist(),
               'rng_python': repr(random.getstate()), 'rng_numpy': repr(np.random.get_state())}
    digest.update(json.dumps(payload, sort_keys=True, default=lambda x: x.item()).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    for model in (sim.action_policy_network, sim.value_network, sim.guide_network):
        for tensor in model.state_dict().values():
            digest.update(tensor.detach().cpu().numpy().tobytes())
    for optimizer in (sim.action_optimizer, sim.value_optimizer, sim.guide_optimizer):
        for parameter in optimizer.param_groups[0]['params']:
            for key, value in sorted(optimizer.state.get(parameter, {}).items()):
                digest.update(key.encode())
                digest.update(value.detach().cpu().numpy().tobytes() if torch.is_tensor(value)
                              else repr(value).encode())
    return digest.hexdigest()


def usage_snapshot():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {name: getattr(usage, name) for name in
            ('ru_utime', 'ru_stime', 'ru_minflt', 'ru_majflt', 'ru_nvcsw', 'ru_nivcsw')}


def profile_summary(profiler, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(destination))
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        try:
            filename = str(Path(filename).resolve().relative_to(root)) if filename != '~' else '~'
        except ValueError:
            pass
        rows.append({'file': filename, 'line': line, 'function': function,
                     'primitive_calls': primitive, 'calls': calls,
                     'self_seconds': own, 'cumulative_seconds': cumulative})
    hotspot_names = ('get_zeitgeist_multiplier', 'get_global_path_distribution', 'mark_paths_dirty',
                     'dominant_path_idx', 'select_opponents', 'entity_interaction',
                     'evaluate_entities', 'concepts')
    return {'raw_stats': str(destination), 'total_self_seconds': stats.total_tt,
            'known_incomplete': True, 'rejected_for_percentages': True,
            'known_issue': 'https://github.com/pybind/pybind11/issues/5600',
            'integrity_violations': [row for row in rows
                                     if row['self_seconds'] > row['cumulative_seconds'] + 1e-9],
            'total_calls': stats.total_calls, 'primitive_calls': stats.prim_calls,
            'top_self': sorted(rows, key=lambda row: row['self_seconds'], reverse=True)[:50],
            'top_cumulative': sorted(rows, key=lambda row: row['cumulative_seconds'], reverse=True)[:50],
            'hotspots': [row for row in rows if any(name in row['function'] for name in hotspot_names)],
            'functions': rows,
            'scope': 'cProfile instruments the calling Python thread only. Native calls include elapsed '
                     'call/wait time, not worker-thread attribution. Self time includes native calls; '
                     'cumulative times overlap and must not be added. Profiling overhead is not '
                     'subtracted and can disproportionately inflate short Python calls.'}


class BoundaryTimers:
    """Explicit nested spans; independent of interpreter/C-extension profile events."""

    def __init__(self):
        self.stack = []
        self.rows = {}
        self.patches = []
        self.native_encounter_entry = False
        self.native_encounter_successes = 0

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*positional, **keywords):
            start = time.perf_counter()
            frame = [name, 0.0]
            self.stack.append(frame)
            try:
                return function(*positional, **keywords)
            finally:
                elapsed = time.perf_counter() - start
                self.stack.pop()
                row = self.rows.setdefault(name, {'name': name, 'calls': 0,
                                                  'inclusive_seconds': 0.0,
                                                  'exclusive_instrumented_seconds': 0.0})
                row['calls'] += 1
                row['inclusive_seconds'] += elapsed
                row['exclusive_instrumented_seconds'] += elapsed - frame[1]
                if self.stack:
                    self.stack[-1][1] += elapsed
        return measured

    def replace(self, owner, name, value):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def install(self):
        from entities import Pathstrider
        from interaction_handler import InteractionHandler
        from parliament_manager import ParliamentManager
        from population_manager import PopulationManager
        from aeonic_cycle_manager import AeonicCycleManager
        simulation_module = sys.modules[AeonEvolution.__module__]
        self.native_encounter_entry = hasattr(simulation_module, 'try_run_encounters')
        targets = [
            (AeonEvolution, ('_run_one_generation', '_evolve_and_grow', '_train_hybrid_guide_network')),
            (AeonicCycleManager, ('run_aeonic_cycle_generation', 'aeonic_interaction')),
        ]
        if not self.native_encounter_entry:
            # A1's fine-grained spans are retained for its original loop. The
            # new engine deliberately rejects modified handler/property methods.
            targets.extend([
                (InteractionHandler, ('select_opponents', 'entity_interaction', 'update_bins')),
                (PopulationManager, ('get_global_path_distribution', 'mark_paths_dirty',
                                     '_add_new_entities', 'recalculate_and_normalize_entity')),
                (ParliamentManager, ('get_zeitgeist_multiplier', 'hold_election')),
            ])
        for owner, names in targets:
            for name in names:
                original = getattr(owner, name)
                self.replace(owner, name, self.wrap(f'{owner.__name__}.{name}', original))
        if not self.native_encounter_entry:
            original_property = Pathstrider.dominant_path_idx
            self.replace(Pathstrider, 'dominant_path_idx', property(
                self.wrap('Pathstrider.dominant_path_idx', original_property.fget),
                original_property.fset, original_property.fdel, original_property.__doc__))
        else:
            import native_encounters as encounter_module
            self.replace(encounter_module, '_eligible', self.wrap(
                'encounters.eligibility', encounter_module._eligible))
            original_entry_points = encounter_module._entry_points

            @functools.wraps(original_entry_points)
            def observed_entry_points(library):
                run, error = original_entry_points(library)
                return self.wrap('encounters.C++_run', run), error

            self.replace(encounter_module, '_entry_points', observed_entry_points)
            original_encounters = simulation_module.try_run_encounters
            @functools.wraps(original_encounters)
            def observed_encounters(*positional, **keywords):
                result = original_encounters(*positional, **keywords)
                if result is not None:
                    self.native_encounter_successes += 1
                return result
            self.replace(simulation_module, 'try_run_encounters', self.wrap(
                'try_run_encounters', observed_encounters))
        self.replace(simulation_module, 'evaluate_entities', self.wrap(
            'evaluate_entities', simulation_module.evaluate_entities))
        original_evaluator = AeonEvolution._get_native_evaluator
        tracker = self

        class EvaluatorProxy:
            def __init__(self, target):
                self.target = target
                self.concepts = tracker.wrap('native.BatchEvaluator.concepts', target.concepts)

            def __getattr__(self, name):
                return getattr(self.target, name)

        @functools.wraps(original_evaluator)
        def get_evaluator(sim):
            evaluator = original_evaluator(sim)
            return None if evaluator is None else EvaluatorProxy(evaluator)

        self.replace(AeonEvolution, '_get_native_evaluator', get_evaluator)

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def summary(self, phase, generations):
        errors = []
        def count(name):
            return self.rows.get(name, {}).get('calls', 0)
        if self.stack:
            errors.append('Unclosed boundary spans')
        if phase == 'human':
            for name in ('AeonEvolution._run_one_generation', 'AeonEvolution._evolve_and_grow'):
                if count(name) != generations:
                    errors.append(f'{name}: expected {generations}, got {count(name)}')
            if self.native_encounter_entry:
                if count('try_run_encounters') != generations or self.native_encounter_successes != generations:
                    errors.append('Native encounter entry must execute successfully once per human generation')
            elif count('PopulationManager.get_global_path_distribution') < count('InteractionHandler.entity_interaction'):
                errors.append('Distribution call count is smaller than interaction count')
        elif count('AeonicCycleManager.run_aeonic_cycle_generation') != generations:
            errors.append('Aeonic generation span count differs from generation count')
        if any(row['exclusive_instrumented_seconds'] < -1e-9 for row in self.rows.values()):
            errors.append('Negative exclusive instrumented time')
        if self.native_encounter_entry and 'try_run_encounters' in self.rows:
            self.rows['try_run_encounters']['successful_returns'] = self.native_encounter_successes
        return {'valid': not errors, 'integrity_errors': errors,
                'native_encounter_entry_present': self.native_encounter_entry,
                'native_encounter_successful_calls': self.native_encounter_successes,
                'spans': sorted(self.rows.values(), key=lambda row: row['inclusive_seconds'], reverse=True),
                'scope': 'Explicit Python wrappers count calls and perf_counter spans without cProfile. '
                         'Inclusive spans overlap. Exclusive instrumented time subtracts only wrapped '
                         'descendants and still includes unwrapped Python/native work and observer overhead; '
                         'it is not pure Python interpreter time. Native scoring span includes conversion, '
                         'worker startup/wait and computation. Wrappers are restored before state capture. '
                         'Results must match plain state/RNG digests and call-count integrity gates. '
                         'When the standalone encounter entry is present, protected entity/handler/manager '
                         'methods are left untouched. Only its outer entry is wrapped and every human '
                         'generation must return a non-None native result, ruling out observer-forced fallback. '
                         'The entry span includes eligibility, packing, core execution and writeback. '
                         'Separate eligibility and C ABI execution spans identify marshaling overhead '
                         'without modifying any protected entity or handler method.'}


def sample(population, generations, profile_path=None, boundary_timers=False):
    random.seed(731)
    np.random.seed(731)
    torch.manual_seed(731)
    options = {key: value for key, value in config['simulation'].items()
               if key != 'elites_to_keep_in_cycle'}
    options.update(num_initial_entities=population, population_soft_cap=population,
                   population_hard_cap=2*population, bard_frequency=0, laertes_frequency=0,
                   kaoselanna_llm_enabled=False)
    sim = AeonEvolution(**options)
    boundaries = None
    try:
        sim._create_initial_population(create_reincarnator=args.with_reincarnator)
        sim.legacy_manager.newborn_affinity_modifier = np.zeros(12)
        sim.generation = 176200
        sim.total_generations = 335503360
        if args.phase == 'aeonic':
            sim.population_manager.update_golden_ones(sim.population)
            sim.reincarnator = next((p for p in sim.population if p.titan_aspect == '负世'),
                                   max(sim.population, key=lambda p: p.score))
            sim.reincarnator.trait = 'Reincarnator'
            sim.reincarnator.name = 'Neikos-0496'
            sim.aeonic_cycle_mode = True
            sim.aeonic_cycle_manager.initialize_aeonic_cycle(sim.reincarnator, sim.population,
                                                            sim.cosmic_zeitgeist)
        profiler = cProfile.Profile() if profile_path is not None else None
        boundaries = BoundaryTimers() if boundary_timers else None
        if boundaries is not None:
            try:
                boundaries.install()
            except BaseException:
                boundaries.restore()
                raise
        usage_before = usage_snapshot()
        start_cpu, start = time.process_time(), time.perf_counter()
        if profiler is not None:
            profiler.enable()
        try:
            for generation in range(generations):
                if args.phase == 'human':
                    dead = sim._run_one_generation()
                    sim._evolve_and_grow(dead)
                else:
                    if generation:
                        titans = [p for p in sim.population if p.titan_aspect and p.trait == 'GoldenOne']
                        sim.titan_bosses.clear()
                        for entity in titans:
                            entity.is_titan_boss = True
                            entity.trait = 'Mortal'
                            entity.hp = 250.0 + entity.score
                            sim.titan_bosses.append(entity)
                        sim.population_manager.update_golden_ones(sim.population)
                    manager = sim.aeonic_cycle_manager
                    manager.run_aeonic_cycle_generation(sim.population, sim.reincarnator, sim.cosmic_zeitgeist)
                    if len(sim.reincarnator.held_fire_seeds) >= len(TITAN_NAMES):
                        end = manager.end_aeonic_cycle(sim.reincarnator, sim.population, sim.base_titan_affinities,
                                                      sim.population_manager.mutation_rate, sim.population_soft_cap,
                                                      sim.cosmic_zeitgeist, config['simulation']['elites_to_keep_in_cycle'])
                        if end:
                            raise RuntimeError('Benchmark ended early; reduce its fixed generation count')
                sim.generation += 1
        finally:
            if profiler is not None:
                profiler.disable()
        seconds = time.perf_counter() - start
        cpu_seconds = time.process_time() - start_cpu
        usage_after = usage_snapshot()
        if boundaries is not None:
            boundaries.restore()
        usage_delta = ({name: usage_after[name] - usage_before[name] for name in usage_before}
                       if usage_before is not None else None)
        steps = [int(state['step'].item()) for optimizer in (sim.action_optimizer, sim.value_optimizer)
                 for state in optimizer.state.values()]
        guide_steps = [int(state['step'].item()) for state in sim.guide_optimizer.state.values()]
        if args.phase == 'aeonic' and (not steps or min(steps) <= 0):
            raise AssertionError('Aeonic benchmark must perform real optimizer updates')
        result = {'seconds': seconds, 'cpu_seconds': cpu_seconds, 'population': len(sim.population),
                  'state_digest': digest_state(sim), 'optimizer_steps': steps,
                  'guide_optimizer_steps': guide_steps,
                  'resource_delta': usage_delta}
        if profiler is not None:
            result['profile'] = profile_summary(profiler, profile_path)
        if boundaries is not None:
            result['boundaries'] = boundaries.summary(args.phase, generations)
        return result
    finally:
        if boundaries is not None:
            boundaries.restore()
        if hasattr(sim, 'close'):
            sim.close()
        else:
            sim.debugger.close()


sample(max(100 if args.phase == 'aeonic' else 20, min(args.population, 100)), 2)
runs = [sample(args.population, args.generations,
               args.profile_output_dir/f'sample-{index}.prof' if args.profile_output_dir else None,
               boundary_timers=args.boundary_timers)
        for index in range(args.repeats)]
result = {'source_root': str(root),
          'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
          'platform': platform.platform(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count(),
          'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
          'torch_threads': 1, 'native_threads_requested': args.native_threads,
          'native_min_rows': args.native_min_rows, 'phase': args.phase,
          'profiled': args.profile_output_dir is not None,
          'boundary_timers': args.boundary_timers,
          'initial_reincarnator': args.with_reincarnator,
          'initial_population': args.population, 'generations': args.generations,
          'runs': runs, 'median_seconds': statistics.median(r['seconds'] for r in runs),
          'scope': 'One world; native CPU; seed 731; no LLM/log/console/export; '
                   'initial_reincarnator records whether original seeded selection was requested; '
                   'aeonic has real ordered online training. Setup/warmup/state capture excluded. '
                   'Each measured simulation is fresh; first native pool startup is timed if reached. '
                   'Aeonic does not exercise the human batch-scoring pool. '
                   'Not a complete-world or LLM latency measurement.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
if any(not run.get('boundaries', {}).get('valid', True) for run in runs):
    raise SystemExit('Boundary observer call-count/timing integrity failed; inspect result JSON')
