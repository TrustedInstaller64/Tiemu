"""Bounded cloud diagnosis of the A1 -> A2 aeonic timing regression.

This is a diagnostic runner, not a product multi-world scheduler. The parent
starts isolated, sequential workers with rotated order. Workers use real ordered
aeonic generations and Adam updates. Observer runs are separate from timing
runs. Optional distribution/GC ablations exist only in the worker process.
"""
import argparse
import ast
import functools
import gc
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
import types


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--generations', type=int, default=2)
    parser.add_argument('--mode', choices=('plain', 'legacy-distribution', 'no-gc',
                                           'materialize-dicts', 'legacy-path-access'), default='plain')
    parser.add_argument('--observe', action='store_true')
    args = parser.parse_args()
    if min(args.blocks, args.samples, args.generations) < 1:
        parser.error('counts must be positive')
    if args.worker and args.source_root is None:
        parser.error('--worker requires --source-root')
    if not args.worker and args.candidate is None:
        parser.error('the parent requires --candidate')
    return args


class Timers:
    """Independent inclusive spans, without cProfile's extension frame events."""
    def __init__(self):
        self.rows = {}
        self.restore_actions = []
        self.stack = []

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*positional, **keywords):
            frame = [0.0]
            self.stack.append(frame)
            start = time.perf_counter()
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
                row['exclusive_instrumented_seconds'] += elapsed - frame[0]
                if self.stack:
                    self.stack[-1][0] += elapsed
        return measured

    def replace(self, owner, name, value):
        # Class methods inherit into instances; restore exact prior ownership.
        owned = name in vars(owner)
        original = getattr(owner, name)
        setattr(owner, name, value)
        self.restore_actions.append((owner, name, owned, original))

    def install(self, sim, native, torch):
        from aeonic_cycle_manager import AeonicCycleManager
        from entities import Pathstrider
        from population_manager import PopulationManager
        from parliament_manager import ParliamentManager
        for owner, names in (
            (AeonicCycleManager, ('run_aeonic_cycle_generation', 'select_aeonic_opponents',
                                 'initialize_aeonic_cycle', 'end_aeonic_cycle')),
            (PopulationManager, ('get_global_path_distribution', 'recalculate_and_normalize_entity',
                                 'update_golden_ones', '_add_new_entities',
                                 'check_and_replenish_population')),
            (Pathstrider, ('recalculate_concepts',)),
            (ParliamentManager, ('get_zeitgeist_multiplier',)),
        ):
            for name in names:
                self.replace(owner, name, self.wrap(f'{owner.__name__}.{name}', getattr(owner, name)))
        original_interaction = AeonicCycleManager.aeonic_interaction
        training = self.wrap('interaction.reincarnator', original_interaction)
        ordinary = self.wrap('interaction.other', original_interaction)

        @functools.wraps(original_interaction)
        def interaction(manager, attacker, defender, reincarnator, *positional, **keywords):
            target = training if attacker is reincarnator else ordinary
            return target(manager, attacker, defender, reincarnator, *positional, **keywords)

        self.replace(AeonicCycleManager, 'aeonic_interaction', interaction)
        for label, model in (('actor.forward', sim.action_policy_network),
                             ('value.forward', sim.value_network)):
            self.replace(model, 'forward', self.wrap(label, model.forward))
        for label, optimizer in (('actor', sim.action_optimizer), ('value', sim.value_optimizer)):
            for name in ('zero_grad', 'step'):
                self.replace(optimizer, name, self.wrap(f'{label}.{name}', getattr(optimizer, name)))
        self.replace(torch.Tensor, 'backward', self.wrap('autograd.backward', torch.Tensor.backward))
        for name in ('path_affinities', 'concepts', 'normalize_pipeline', 'normalize', 'purify'):
            self.replace(native, name, self.wrap(f'native.{name}', getattr(native, name)))
        # Do not wrap lazy path properties or TitanToPathModel.get_path_affinities:
        # their identity is part of A2's batch-access eligibility contract.

    def restore(self):
        for owner, name, owned, original in reversed(self.restore_actions):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.restore_actions.clear()

    def summary(self, generations, updates):
        def count(name):
            return self.rows.get(name, {}).get('calls', 0)
        errors = []
        if self.stack:
            errors.append('unclosed spans')
        if count('AeonicCycleManager.run_aeonic_cycle_generation') != generations:
            errors.append('generation count differs')
        if count('actor.step') != updates or count('value.step') != updates:
            errors.append('observed optimizer steps differ from actual Adam state')
        if count('autograd.backward') != 2 * updates:
            errors.append('backward count differs from the two ordered updates')
        if any(row['exclusive_instrumented_seconds'] < -1e-8 for row in self.rows.values()):
            errors.append('negative exclusive span')
        return {'valid': not errors, 'errors': errors,
                'spans': sorted(self.rows.values(), key=lambda item: item['inclusive_seconds'], reverse=True),
                'scope': 'Inclusive spans overlap. Exclusive values subtract only observed descendants; '
                         'they include unobserved native/Python work and observer overhead. Observers '
                         'are never used for speedup estimates. Lazy-path/model identities are untouched. '
                         'Plain/observed full checked-state digests must match.'}


def worker(args):
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1')
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
                             native_threads=1, native_min_rows=4096)
    native = configure_backend('cpp')
    core_path = Path(getattr(native, '__core_file__', native.__file__)).resolve()
    if Path(native.__file__).resolve().parent != root or core_path.parent != root:
        raise RuntimeError('the native backend must come from the selected in-place checkout')
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    class QuietDebugger:
        def __init__(self, *positional, **keywords):
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
    import population_manager as population_module
    if Path(sys.modules['simulation'].__file__).resolve().parent != root:
        raise RuntimeError('the simulation came from the wrong checkout')

    if args.mode == 'legacy-distribution':
        source = (args.baseline.resolve()/'population_manager.py').read_text(encoding='utf-8')
        cls = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == 'PopulationManager')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'get_global_path_distribution')
        namespace = dict(vars(population_module))
        exec(compile(ast.Module(body=[method], type_ignores=[]),
                     'frozen-A1:get_global_path_distribution', 'exec'), namespace)
        population_module.PopulationManager.get_global_path_distribution = namespace[method.name]

    patched_path_consumers = []
    if args.mode == 'legacy-path-access':
        # A2 inspects entity.__dict__ in the batch helper. CPython's managed
        # instance layout may then affect later, otherwise unchanged attribute
        # scans. Disable every imported helper alias before world creation,
        # including the first election; replacing only distribution misses it.
        def use_original_getters(*positional, **keywords):
            return None
        for name in ('parliament_manager', 'population_manager',
                     'diversity_manager', 'stagnation_manager'):
            module = importlib.import_module(name)
            if not hasattr(module, '_native_dominant_paths'):
                raise AssertionError(f'{name}: expected A2 batch-path alias is absent')
            module._native_dominant_paths = use_original_getters
            patched_path_consumers.append(name)

    def digest_checked_state(sim):
        """Hash raw arrays/RNG (never truncated reprs) and ordered state references."""
        digest = hashlib.sha256()
        def add(value):
            if torch.is_tensor(value):
                add(value.detach().cpu().numpy())
            elif isinstance(value, np.ndarray):
                digest.update(b'array')
                add(value.dtype.str)
                add(value.shape)
                digest.update(value.tobytes(order='C'))
            elif isinstance(value, np.generic):
                # Serialize scalar values consistently across native adapters;
                # exact scalar types/aliases are covered by functional tests.
                add(value.item())
            elif isinstance(value, dict):
                digest.update(b'dict')
                for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                    add(key)
                    add(value[key])
            elif isinstance(value, (tuple, list)):
                digest.update(b'sequence')
                add(len(value))
                for item in value:
                    add(item)
            elif isinstance(value, set):
                add(sorted(value))
            elif isinstance(value, float):
                digest.update(b'float' + struct.pack('!d', value))
            elif value is None or isinstance(value, (str, int, bool)):
                encoded = json.dumps(value, ensure_ascii=True).encode('ascii')
                digest.update(type(value).__name__.encode() + str(len(encoded)).encode() + b':' + encoded)
            else:
                raise TypeError(f'unhandled checked-state value: {type(value).__name__}')

        entities = []
        references = {}
        def reference(entity):
            if entity is None:
                return None
            if id(entity) not in references:
                references[id(entity)] = len(entities)
                entities.append(entity)
            return references[id(entity)]
        manager = sim.aeonic_cycle_manager
        payload = {
            'population': [reference(entity) for entity in sim.population],
            'reincarnator': reference(sim.reincarnator),
            'bosses': [reference(entity) for entity in sim.titan_bosses],
            'names': {name: reference(entity) for name, entity in sorted(sim.name_to_entity_map.items())},
            'existing_names': sim.existing_names,
            'titan_entities': {name: reference(entity) for name, entity in sorted(manager.titan_entities.items())},
            'cycle': manager.aeonic_cycle_number,
            'defeated': manager.defeated_titans_this_cycle,
            'destruction_score': manager.baie_destruction_score,
            'seeds_collected': manager.baie_total_seeds_collected,
            'generation': sim.generation,
            'blueprint': sim.base_titan_affinities,
            'cosmic_zeitgeist': sim.cosmic_zeitgeist,
            'cosmic_tide': sim.population_manager.cosmic_tide_vector,
            'mapping_matrix': sim.titan_to_path_model_instance.titan_to_path_matrix,
            'feedback_matrix': sim.titan_to_path_model_instance.path_to_titan_feedback_matrix,
            'mutation_rate': sim.population_manager.mutation_rate,
            'seats': sim.parliament_manager.seats,
            'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
        }
        payload['entities'] = [
            {name: value for name, value in vars(entity).items() if name != '_titan_to_path_model'}
            for entity in entities]
        payload['networks'] = [network.state_dict() for network in (
            sim.action_policy_network, sim.value_network, sim.guide_network)]
        payload['optimizers'] = [optimizer.state_dict() for optimizer in (
            sim.action_optimizer, sim.value_optimizer, sim.guide_optimizer)]
        add(payload)
        whole = digest.hexdigest()
        components = {}
        for key, value in payload.items():
            digest = hashlib.sha256()
            add(value)
            components[key] = digest.hexdigest()
        return whole, components

    def sample(generations, observe=False):
        random.seed(731)
        np.random.seed(731)
        torch.manual_seed(731)
        options = {key: value for key, value in config['simulation'].items()
                   if key != 'elites_to_keep_in_cycle'}
        options.update(num_initial_entities=200, population_soft_cap=200,
                       population_hard_cap=400, bard_frequency=0, laertes_frequency=0,
                       kaoselanna_llm_enabled=False)
        sim = AeonEvolution(**options)
        timers = None
        original_gc = gc.isenabled()
        try:
            sim._create_initial_population(create_reincarnator=False)
            sim.legacy_manager.newborn_affinity_modifier = np.zeros(12)
            sim.generation = 176200
            sim.total_generations = 335503360
            sim.population_manager.update_golden_ones(sim.population)
            sim.reincarnator = next((entity for entity in sim.population if entity.titan_aspect == '负世'),
                                   max(sim.population, key=lambda entity: entity.score))
            sim.reincarnator.trait = 'Reincarnator'
            sim.reincarnator.name = 'Neikos-0496'
            sim.aeonic_cycle_mode = True
            manager = sim.aeonic_cycle_manager
            if not manager.initialize_aeonic_cycle(sim.reincarnator, sim.population, sim.cosmic_zeitgeist):
                raise AssertionError('aeonic setup did not initialize')
            materialized = 0
            if args.mode == 'materialize-dicts':
                # Apply the hypothesized A2 layout side effect to A1 after all
                # setup, without changing a field, RNG draw, method or algorithm.
                # Do not inspect dicts in any other timing mode before timing.
                for entity in sim.population:
                    entity.__dict__
                    materialized += 1
            if observe:
                timers = Timers()
                timers.install(sim, native, torch)
            # Keep normal Python GC for reported plain timings. The separate
            # no-gc ablation disables it only over the measured phase loop.
            gc_before = gc.get_stats()
            if args.mode == 'no-gc':
                gc.disable()
            start_cpu, start = time.process_time(), time.perf_counter()
            for generation in range(generations):
                if generation:
                    titans = [entity for entity in sim.population
                              if entity.titan_aspect and entity.trait == 'GoldenOne']
                    sim.titan_bosses.clear()
                    for entity in titans:
                        entity.is_titan_boss = True
                        entity.trait = 'Mortal'
                        entity.hp = 250.0 + entity.score
                        sim.titan_bosses.append(entity)
                    sim.population_manager.update_golden_ones(sim.population)
                manager.run_aeonic_cycle_generation(sim.population, sim.reincarnator, sim.cosmic_zeitgeist)
                if len(sim.reincarnator.held_fire_seeds) >= len(TITAN_NAMES):
                    ended = manager.end_aeonic_cycle(
                        sim.reincarnator, sim.population, sim.base_titan_affinities,
                        sim.population_manager.mutation_rate, sim.population_soft_cap,
                        sim.cosmic_zeitgeist, config['simulation']['elites_to_keep_in_cycle'])
                    if ended:
                        raise AssertionError('the real phase ended before the requested fixed workload')
                sim.generation += 1
            seconds = time.perf_counter() - start
            cpu_seconds = time.process_time() - start_cpu
            if original_gc:
                gc.enable()
            gc_after = gc.get_stats()
            if timers:
                timers.restore()
            step_groups = [[int(state['step'].item()) for state in optimizer.state.values()]
                           for optimizer in (sim.action_optimizer, sim.value_optimizer)]
            if not all(group and len(set(group)) == 1 and group[0] > 0 for group in step_groups):
                raise AssertionError('both actual Adam optimizers must update every parameter')
            if step_groups[0][0] != step_groups[1][0]:
                raise AssertionError('the two ordered optimizers performed different update counts')
            state_digest, components = digest_checked_state(sim)
            result = {'seconds': seconds, 'cpu_seconds': cpu_seconds,
                      'state_digest': state_digest, 'state_components': components,
                      'optimizer_updates': step_groups[0][0], 'population': len(sim.population),
                      'setup_entity_dicts_explicitly_materialized': materialized,
                      'cycle': manager.aeonic_cycle_number,
                      'gc_collections': [after['collections'] - before['collections']
                                         for before, after in zip(gc_before, gc_after)]}
            if timers:
                result['observation'] = timers.summary(generations, step_groups[0][0])
                if not result['observation']['valid']:
                    raise AssertionError(f'observer integrity failed: {result["observation"]}')
            return result
        finally:
            if original_gc:
                gc.enable()
            if timers:
                timers.restore()
            sim.close()

    # Separate fresh warmup worlds, never reuse weights/world state in samples.
    sample(2)
    sample(2)
    runs = [sample(args.generations, args.observe) for _ in range(args.samples)]
    report = {
        'root': str(root),
        'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'core_file': str(core_path), 'binding': getattr(native, '__binding__', 'pybind-baseline'),
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
        'mode': args.mode, 'observed': args.observe, 'population': 200,
        'batch_path_aliases_disabled_before_setup': patched_path_consumers,
        'generations': args.generations, 'runs': runs,
        'median_seconds': statistics.median(run['seconds'] for run in runs),
        'scope': 'CPU Torch/BLAS/native 1 thread; fixed seed 731; real aeonic phase and ordered '
                 'online Adam updates. Fresh simulation for every sample; setup, two separate '
                 'warmup worlds, state hashing and teardown excluded. No LLM/log/display/export. '
                 'GC uses the interpreter default except the explicitly labeled no-gc ablation. '
                 'State includes population/reference order, entity raw fields/caches, name maps, '
                 'cycle counters/seeds, blueprint/seats, complete three network/optimizer states '
                 'and untruncated Python/NumPy/Torch RNG. Not arbitrary full-program equivalence.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'worker.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


def parent(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    configurations = [('A1', baseline, 'plain'), ('A2', candidate, 'plain'),
                      ('A2-legacy-distribution', candidate, 'legacy-distribution'),
                      ('A1-no-gc', baseline, 'no-gc'), ('A2-no-gc', candidate, 'no-gc'),
                      ('A1-materialized-dicts', baseline, 'materialize-dicts'),
                      ('A2-legacy-path-access', candidate, 'legacy-path-access')]
    reports = []
    for generations in (2, 12):
        runs = {name: [] for name, _, _ in configurations}
        def execute(name, root, mode, block, observe=False):
            destination = output/f'g{generations}-{name}-{block}'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--source-root', str(root), '--baseline', str(baseline),
                       '--output-dir', str(destination), '--generations', str(generations),
                       '--samples', '1' if observe else str(args.samples), '--mode', mode]
            if observe:
                command.append('--observe')
            subprocess.run(command, check=True, timeout=240,
                           env={**os.environ, 'PYTHONHASHSEED': '0'}, stdout=subprocess.DEVNULL)
            return json.loads((destination/'worker.json').read_text(encoding='utf-8'))
        for block in range(args.blocks):
            rotation = block % len(configurations)
            order = configurations[rotation:] + configurations[:rotation]
            if block % 2:
                order = list(reversed(order))
            for name, root, mode in order:
                runs[name].append(execute(name, root, mode, block))
        observations = {name: execute(name, root, mode, 'observed', True)
                        for name, root, mode in configurations if mode != 'no-gc'}
        digests = {run['state_digest'] for series in runs.values() for record in series for run in record['runs']}
        digests.update(run['state_digest'] for record in observations.values() for run in record['runs'])
        if len(digests) != 1:
            raise AssertionError(f'g{generations}: timing/ablation/observer checked states diverged; inspect files')
        process_medians = {name: [record['median_seconds'] for record in series] for name, series in runs.items()}
        medians = {name: statistics.median(values) for name, values in process_medians.items()}
        row = {'population': 200, 'generations': generations,
               'blocks': args.blocks, 'samples_per_block': args.samples,
               'process_medians_seconds': process_medians, 'median_seconds': medians,
               'speedup_over_A1': {name: medians['A1']/value for name, value in medians.items()},
               'per_block_A2_over_A1_seconds': [b/a for a, b in zip(process_medians['A1'], process_medians['A2'])],
               'ablation_wall_ratios': {
                   'A1_materialized_dicts_over_A1': medians['A1-materialized-dicts']/medians['A1'],
                   'A2_legacy_path_access_over_A2': medians['A2-legacy-path-access']/medians['A2'],
               },
               'same_checked_state_and_rng': True,
               'optimizer_updates': observations['A2']['runs'][0]['optimizer_updates'],
               'observations': {name: record['runs'][0]['observation'] for name, record in observations.items()}}
        reports.append(row)
        print(json.dumps(row), flush=True)
    report = {'baseline': str(baseline), 'candidate': str(candidate), 'cases': reports,
              'scope': 'Frozen A1/A2 on the same hosted runner; sequential isolated processes, '
                       'rotated/reversed process order, within-process fresh-world repeats. Median '
                       'of process medians retains process blocks as independent samples. All raw '
                       'samples, GC counts, environment and state digests are retained. Legacy '
                       'distribution replaces only the frozen A1 method inside the A2 diagnostic '
                       'process. No-gc disables collection only within the measured phase. A1 '
                       'materialized-dicts touches every initial entity.__dict__ after setup; A2 '
                       'legacy-path-access disables every imported batch-path helper alias before '
                       'initialization/election. Ablations change no production code or numerical '
                       'rule. The longer phase may add candidate selections without adding optimizer '
                       'updates; actual update and observed selection counts are reported. These '
                       'short and longer finite-stage workloads '
                       'are not complete-world, local M1, power or multicore measurements.'}
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    options = arguments()
    worker(options) if options.worker else parent(options)
