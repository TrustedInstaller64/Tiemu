"""Cloud-only comparison of the current and frozen original generation loops.

Both alternatives use the same checkout, C ABI and unchanged interaction and
training methods. Full-cycle fixtures retain the existing reincarnator reference
behavior. A separately labelled controlled fixture keeps training active without
cycle transitions; its timings are not complete-world throughput measurements.
"""
import argparse
import ast
import gc
import hashlib
import itertools
import json
import logging
import math
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


REFERENCE = 'f2073f001af4bc3f03da6d96d5360aa9a2027e0a'
CONFIGURATIONS = ('current', 'legacy-generation', 'current-repeat')
CASES = tuple((mode, population, generations)
              for mode in ('full-cycle', 'controlled-training')
              for population in (200, 1000) for generations in (2, 12))
CONTROL_DESCRIPTION = (
    'Generation-only diagnostic: one live reincarnator is the only preferred '
    'attacker; all other initial entities are Mortal with no titan form or seeds. '
    'Initial affinities are small finite vectors and scores are 2..8. The actor '
    'output layer initially has zero weights and biases [3,-3], favoring '
    'negotiation without fixing sampled actions. Real Categorical sampling, '
    'backward and ordered Adam updates remain unchanged. No golden promotion, '
    'end-cycle transition, target replenishment or per-generation reset is added. '
    'Population may decline according to the original interaction rules. This '
    'is not a complete aeonic cycle or a production initial policy.')


def case_name(case):
    return '%s-%s-g%s' % case


def initialize(args):
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
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
    core = Path(getattr(native, '__core_file__', native.__file__)).resolve()
    if Path(native.__file__).resolve().parent != root or core.parent != root:
        raise RuntimeError('The Python bridge and C ABI must come from --source-root')
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    class QuietDebugger:
        def __init__(self, *args, **kwargs):
            self.paused, self.stop_requested, self.last_command = False, False, ''
        def poll(self): pass
        def close(self): pass

    stub = types.ModuleType('debugger')
    stub.Debugger = QuietDebugger
    sys.modules['debugger'] = stub
    from simulation import AeonEvolution
    from constants import TITAN_NAMES
    import aeonic_cycle_manager as cycle_module
    import native_aeonic
    import native_lifecycle
    if Path(sys.modules['simulation'].__file__).resolve().parent != root:
        raise RuntimeError('The simulation must come from --source-root')
    source = subprocess.check_output(
        ['git', '-C', str(root), 'show', f'{REFERENCE}:aeonic_cycle_manager.py'], encoding='utf-8')
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'AeonicCycleManager')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'run_aeonic_cycle_generation')
    scope = dict(vars(cycle_module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 'frozen-A3-generation', 'exec'), scope)
    original_selector = cycle_module.AeonicCycleManager.select_aeonic_opponents
    original_interaction = cycle_module.AeonicCycleManager.aeonic_interaction
    if args.configuration == 'legacy-generation':
        cycle_module.AeonicCycleManager.run_aeonic_cycle_generation = scope[method.name]
    assert cycle_module.AeonicCycleManager.select_aeonic_opponents is original_selector
    assert cycle_module.AeonicCycleManager.aeonic_interaction is original_interaction
    metadata = {
        'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                                 encoding='utf-8').strip(),
        'reference_generation_commit': REFERENCE, 'python': platform.python_version(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'logical_cpus': os.cpu_count(), 'numpy': np.__version__, 'torch': torch.__version__,
        'binding': getattr(native, '__binding__', None), 'core_file': str(core),
        'torch_threads': torch.get_num_threads(), 'native_threads': 1,
        'golden_enabled': native_lifecycle.NATIVE_GOLDEN_ENABLED,
        'minimum_index_population': native_aeonic.MIN_INDEX_POPULATION,
        'index_with_reincarnator': native_aeonic.INDEX_WITH_REINCARNATOR,
    }
    if platform.system() == 'Darwin':
        for key in ('machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.memsize'):
            result = subprocess.run(['sysctl', '-n', key], capture_output=True,
                                    text=True, encoding='utf-8', timeout=5)
            metadata[key] = result.stdout.strip() if result.returncode == 0 else None
    return types.SimpleNamespace(np=np, torch=torch, config=config, native=native,
                                 AeonEvolution=AeonEvolution, TITAN_NAMES=TITAN_NAMES,
                                 cycle_module=cycle_module, native_aeonic=native_aeonic,
                                 metadata=metadata)


def hash_value(value, runtime):
    """Raw numeric/RNG bytes; never NumPy repr, rounded text or truncated arrays."""
    np, torch = runtime.np, runtime.torch
    digest = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            digest.update(b'tensor')
            add(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(b'array')
            add(item.dtype.str)
            add(item.shape)
            digest.update(item.tobytes(order='C'))
        elif isinstance(item, np.generic):
            digest.update(b'numpy-scalar')
            add(item.dtype.str)
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            add(len(item))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            add(len(item))
            for member in item:
                add(member)
        elif isinstance(item, set):
            digest.update(b'set')
            add(sorted(item, key=lambda member: (type(member).__name__, repr(member))))
        elif type(item) is float:
            digest.update(b'float' + struct.pack('!d', item))
        elif item is None or type(item) in (str, int, bool):
            data = json.dumps(item, ensure_ascii=True).encode('ascii')
            digest.update(type(item).__name__.encode() + str(len(data)).encode() + b':' + data)
        else:
            raise TypeError(f'Uncovered checked-state value: {type(item).__name__}')

    add(value)
    return digest.hexdigest()


def state_digest(sim, runtime):
    entities, ids = [], {}

    def reference(entity):
        if entity is None:
            return None
        if id(entity) not in ids:
            ids[id(entity)] = len(entities)
            entities.append(entity)
        return ids[id(entity)]

    manager, population_manager = sim.aeonic_cycle_manager, sim.population_manager
    payload = {
        'population': [reference(entity) for entity in sim.population],
        'passed_reincarnator': reference(sim.reincarnator),
        'bosses': [reference(entity) for entity in sim.titan_bosses],
        'name_references': {name: reference(entity) for name, entity in sorted(sim.name_to_entity_map.items())},
        'existing_names': sim.existing_names,
        'titan_references': {name: reference(entity) for name, entity in sorted(manager.titan_entities.items())},
        'cycle': manager.aeonic_cycle_number, 'generation': sim.generation,
        'defeated_titans': manager.defeated_titans_this_cycle,
        'destruction': manager.baie_destruction_score, 'seeds_collected': manager.baie_total_seeds_collected,
        'blueprint': sim.base_titan_affinities, 'zeitgeist': sim.cosmic_zeitgeist,
        'cosmic_tide': population_manager.cosmic_tide_vector,
        'mapping_matrix': sim.titan_to_path_model_instance.titan_to_path_matrix,
        'feedback_matrix': sim.titan_to_path_model_instance.path_to_titan_feedback_matrix,
        'population_parameters': {name: getattr(population_manager, name) for name in
                                  ('mutation_rate', 'purity_factor', 'golden_one_cap',
                                   'golden_one_reversion_prob', 'max_affinity_norm', 'min_norm', 'max_norm')},
        'seats': sim.parliament_manager.seats,
        'python_rng': random.getstate(), 'numpy_rng': runtime.np.random.get_state(),
        'torch_rng': runtime.torch.get_rng_state(),
        'actor': sim.action_policy_network.state_dict(), 'value': sim.value_network.state_dict(),
        'guide': sim.guide_network.state_dict(),
        'actor_optimizer': sim.action_optimizer.state_dict(),
        'value_optimizer': sim.value_optimizer.state_dict(),
        'guide_optimizer': sim.guide_optimizer.state_dict(),
    }
    payload['entities'] = [{key: value for key, value in vars(entity).items()
                            if key != '_titan_to_path_model'} for entity in entities]
    # Check identity sharing for public entity arrays separately from their bytes.
    array_ids, array_aliases = {}, []
    for entity in entities:
        for field in ('titan_affinities', '_path_affinities'):
            array = getattr(entity, field)
            if array is not None and id(array) not in array_ids:
                array_ids[id(array)] = len(array_ids)
            array_aliases.append(None if array is None else array_ids[id(array)])
    payload['entity_array_aliases'] = array_aliases
    components = {key: hash_value(value, runtime) for key, value in payload.items()}
    return {'digest': hash_value(components, runtime), 'components': components}


def optimizer_steps(optimizer):
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if not optimizer.state:
        return 0
    values = [int(optimizer.state[parameter]['step'].item()) for parameter in parameters]
    if len(set(values)) != 1 or values[0] <= 0:
        raise AssertionError('Every optimizer parameter must have the same positive actual step')
    return values[0]


def position(sim):
    return next((index for index, entity in enumerate(sim.population) if entity is sim.reincarnator), None)


def build_world(runtime, case):
    mode, population, _ = case
    np, torch = runtime.np, runtime.torch
    random.seed(731)
    np.random.seed(731)
    torch.manual_seed(731)
    options = {key: value for key, value in runtime.config['simulation'].items()
               if key != 'elites_to_keep_in_cycle'}
    options.update(num_initial_entities=population, population_soft_cap=population,
                   population_hard_cap=2 * population, bard_frequency=0, laertes_frequency=0,
                   kaoselanna_llm_enabled=False)
    sim = runtime.AeonEvolution(**options)
    sim._create_initial_population(create_reincarnator=False)
    sim.legacy_manager.newborn_affinity_modifier = np.zeros(12)
    sim.generation, sim.total_generations, sim.aeonic_cycle_mode = 176200, 335503360, True
    if mode == 'full-cycle':
        sim.population_manager.update_golden_ones(sim.population)
        sim.reincarnator = next((entity for entity in sim.population if entity.titan_aspect == '负世'),
                               max(sim.population, key=lambda entity: entity.score))
        sim.reincarnator.trait, sim.reincarnator.name = 'Reincarnator', 'Neikos-0496'
        if not sim.aeonic_cycle_manager.initialize_aeonic_cycle(
                sim.reincarnator, sim.population, sim.cosmic_zeitgeist):
            raise AssertionError('Full-cycle fixture did not initialize')
    else:
        for index, entity in enumerate(sim.population):
            entity.trait, entity.is_titan_form, entity.titan_aspect = 'Mortal', None, None
            entity.is_titan_boss = False
            entity.held_fire_seeds.clear()
            entity.titan_affinities = np.linspace(.05, .65, 12) + (index % 7) * .005
            entity._path_affinities = None
            entity.score = float(2 + index % 7)
        sim.titan_bosses.clear()
        sim.reincarnator = sim.population[0]
        sim.reincarnator.trait = 'Reincarnator'
        sim.aeonic_cycle_manager.aeonic_cycle_number = 1
        with torch.no_grad():
            sim.action_policy_network.fc3.weight.zero_()
            sim.action_policy_network.fc3.bias.copy_(torch.tensor([3., -3.]))
    return sim


def run_phase(sim, runtime, case, observation, configuration):
    mode, _, generations = case
    records = []
    for generation in range(generations):
        if observation is not None:
            actor_before, value_before = optimizer_steps(sim.action_optimizer), optimizer_steps(sim.value_optimizer)
            before = {'population': len(sim.population), 'passed_reincarnator_position': position(sim),
                      'cycle': sim.aeonic_cycle_manager.aeonic_cycle_number}
            start_index = len(observation)
        if mode == 'full-cycle' and generation:
            titans = [entity for entity in sim.population if entity.titan_aspect and entity.trait == 'GoldenOne']
            sim.titan_bosses.clear()
            for entity in titans:
                entity.is_titan_boss, entity.trait = True, 'Mortal'
                entity.hp = 250.0 + entity.score
                sim.titan_bosses.append(entity)
            sim.population_manager.update_golden_ones(sim.population)
        manager = sim.aeonic_cycle_manager
        manager.run_aeonic_cycle_generation(sim.population, sim.reincarnator, sim.cosmic_zeitgeist)
        if mode == 'full-cycle' and len(sim.reincarnator.held_fire_seeds) >= len(runtime.TITAN_NAMES):
            ended = manager.end_aeonic_cycle(
                sim.reincarnator, sim.population, sim.base_titan_affinities,
                sim.population_manager.mutation_rate, sim.population_soft_cap,
                sim.cosmic_zeitgeist, runtime.config['simulation']['elites_to_keep_in_cycle'])
            if ended:
                raise AssertionError('Full-cycle fixture ended before its fixed generation count')
        sim.generation += 1
        if observation is not None:
            after = {'population': len(sim.population), 'passed_reincarnator_position': position(sim),
                     'cycle': manager.aeonic_cycle_number}
            actor_after, value_after = optimizer_steps(sim.action_optimizer), optimizer_steps(sim.value_optimizer)
            delta = {'actor': actor_after - actor_before, 'value': value_after - value_before}
            if delta['actor'] != delta['value']:
                raise AssertionError('The two original ordered optimizer updates diverged')
            if mode == 'controlled-training' and (before['passed_reincarnator_position'] is None
                    or after['passed_reincarnator_position'] is None or delta['actor'] <= 0):
                raise AssertionError('Controlled training stopped: do not report zero work as a speedup')
            calls = observation[start_index:]
            expected_calls = 0 if configuration == 'legacy-generation' or before['population'] < 2 else 1
            if len(calls) != expected_calls:
                raise AssertionError('Observer did not preserve the expected generation entry route')
            records.append({'generation': generation + 1, 'before': before, 'after': after,
                            'optimizer_step_delta': delta, 'index_calls': calls,
                            'checked_state': state_digest(sim, runtime)})
    return records


def sample(runtime, case, configuration, observe=False):
    sim = build_world(runtime, case)
    original_index = runtime.native_aeonic.try_create_aeonic_index
    calls = [] if observe else None
    if observe:
        def observed(manager, population, reincarnator):
            result = original_index(manager, population, reincarnator)
            expected = len(population) >= runtime.native_aeonic.MIN_INDEX_POPULATION and (
                runtime.native_aeonic.INDEX_WITH_REINCARNATOR
                or not any(entity is reincarnator for entity in population))
            if (result is not None) != expected:
                raise AssertionError('Observation changed native eligibility')
            calls.append({'population': len(population),
                          'passed_reincarnator_in_population': any(entity is reincarnator for entity in population),
                          'admitted': result is not None})
            return result
        runtime.native_aeonic.try_create_aeonic_index = observed
    try:
        actor_initial = hash_value(sim.action_policy_network.state_dict(), runtime)
        value_initial = hash_value(sim.value_network.state_dict(), runtime)
        gc_before = gc.get_stats()
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        observations = run_phase(sim, runtime, case, calls, configuration)
        seconds, cpu_seconds = time.perf_counter() - wall_start, time.process_time() - cpu_start
        gc_after = gc.get_stats()
        steps = {'actor': optimizer_steps(sim.action_optimizer), 'value': optimizer_steps(sim.value_optimizer)}
        if steps['actor'] <= 0 or steps['actor'] != steps['value']:
            raise AssertionError('Both real optimizers must execute matching positive update counts')
        actor_changed = actor_initial != hash_value(sim.action_policy_network.state_dict(), runtime)
        value_changed = value_initial != hash_value(sim.value_network.state_dict(), runtime)
        if not actor_changed or not value_changed:
            raise AssertionError('Both networks must actually learn, not merely count optimizer calls')
        return {'seconds': seconds, 'cpu_seconds': cpu_seconds,
                'gc_collections': [after['collections'] - before['collections']
                                   for before, after in zip(gc_before, gc_after)],
                'optimizer_updates_per_optimizer': steps, 'final_population': len(sim.population),
                'passed_reincarnator_final_position': position(sim),
                'actor_parameters_changed': actor_changed, 'value_parameters_changed': value_changed,
                'checked_state': state_digest(sim, runtime), 'generations_observed': observations}
    finally:
        runtime.native_aeonic.try_create_aeonic_index = original_index
        sim.close()


def worker(args):
    runtime = initialize(args)
    cases = list(reversed(CASES)) if args.reverse_cases else CASES
    results = {}
    record = {'configuration': args.configuration, 'observed': args.observe,
              'environment': runtime.metadata, 'case_order': [case_name(case) for case in cases], 'cases': results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        # Fresh warmup and fresh measured worlds preserve actual first-use costs
        # within each world while excluding imports and process initialization.
        sample(runtime, case, args.configuration)
        runs = [sample(runtime, case, args.configuration, args.observe)
                for _ in range(1 if args.observe else args.samples)]
        results[case_name(case)] = {'runs': runs, 'median_seconds': statistics.median(run['seconds'] for run in runs)}
        # Retain completed cases if a later fixture assertion aborts this worker.
        (args.output_dir/'worker.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')


def spread(values):
    quartiles = statistics.quantiles(values, n=4, method='inclusive')
    return {'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values),
            'q1': quartiles[0], 'q3': quartiles[2]}


def parent(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = {name: [] for name in CONFIGURATIONS}
    orders, observations = [], {}

    def execute(name, block, observe=False):
        destination = output/f'{name}-{block}'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source-root',
                   str(args.source_root.resolve()), '--output-dir', str(destination),
                   '--configuration', name, '--samples', str(args.samples)]
        if observe:
            command.append('--observe')
        if isinstance(block, int) and block % 2:
            command.append('--reverse-cases')
        subprocess.run(command, check=True, timeout=900, stdout=subprocess.DEVNULL,
                       env={**os.environ, 'PYTHONHASHSEED': '0'})
        print(json.dumps({'completed_process': name, 'block': block, 'observed': observe}), flush=True)
        return json.loads((destination/'worker.json').read_text(encoding='utf-8'))

    permutations = list(itertools.permutations(CONFIGURATIONS))
    for block in range(args.blocks):
        order = permutations[block % len(permutations)]
        orders.append(order)
        for name in order:
            runs[name].append(execute(name, block))
    # Observation runs never enter the speedup calculation. No protected entity,
    # selector or interaction method is wrapped; state collection is separate.
    for name in CONFIGURATIONS:
        observations[name] = execute(name, 'observed', True)
    report = {
        'source_root': str(args.source_root.resolve()), 'reference_generation_commit': REFERENCE,
        'blocks': args.blocks, 'samples_per_process_per_case': args.samples, 'process_orders': orders,
        'environments': {name: records[0]['environment'] for name, records in runs.items()},
        'controlled_training_initial_state': CONTROL_DESCRIPTION,
        'scope': 'Same-checkout structural ablation. Only the legacy generation method is replaced. '
                 'Full-cycle cases preserve the existing old-reincarnator-reference behavior and do '
                 'not demonstrate sustained training. Controlled cases exclude cycle transitions '
                 'and are separately labelled. Every six blocks cover all process-order permutations. '
                 'Same-code repeats estimate variability, not another optimization. Normal GC stays '
                 'enabled. Imports, setup, warmup, state capture and teardown are excluded from pure '
                 'timings. Observer timings include instrumentation and must not be used as speedups. '
                 'Raw digests cover public entities/order/array identity sharing, core world fields, '
                 'network weights, complete optimizer state and three RNG states; they do not claim '
                 'coverage of every private manager field, console, external LLM or arbitrary trajectories. '
                 'No local M1 or complete-simulation throughput claim.',
        'cases': []}
    all_records = [record for records in runs.values() for record in records] + list(observations.values())
    if len({record['environment']['source_commit'] for record in all_records}) != 1:
        raise AssertionError('Every configuration must use the exact same project commit')
    print(json.dumps({'metadata': {key: value for key, value in report.items() if key != 'cases'}}), flush=True)
    for case in CASES:
        key = case_name(case)
        all_samples = [run for records in runs.values() for record in records
                       for run in record['cases'][key]['runs']]
        observed_samples = {name: record['cases'][key]['runs'][0] for name, record in observations.items()}
        digests = {run['checked_state']['digest'] for run in [*all_samples, *observed_samples.values()]}
        if len(digests) != 1:
            raise AssertionError(f'{key}: current/legacy/repeat/observer checked states differ; inspect raw workers')
        traces = {name: [(row['before'], row['after'], row['optimizer_step_delta'], row['checked_state'])
                         for row in run['generations_observed']] for name, run in observed_samples.items()}
        if any(trace != traces['current'] for trace in traces.values()):
            raise AssertionError(f'{key}: per-generation checked states or training progress differ')
        medians = {name: [record['cases'][key]['median_seconds'] for record in records]
                   for name, records in runs.items()}
        paired = [old/current for old, current in zip(medians['legacy-generation'], medians['current'])]
        repeat_ratios = [repeat/current for repeat, current in zip(medians['current-repeat'], medians['current'])]
        row = {
            'workload': case[0], 'initial_population': case[1], 'generations': case[2],
            'process_medians_seconds': medians,
            'process_median_spread': {name: spread(values) for name, values in medians.items()},
            'legacy_seconds_over_current_seconds': paired,
            'legacy_over_current_ratio_spread': spread(paired),
            'geometric_mean_legacy_over_current': math.exp(statistics.mean(math.log(value) for value in paired)),
            'current_repeat_seconds_over_current_seconds': repeat_ratios,
            'same_code_repeat_ratio_spread': spread(repeat_ratios),
            'same_checked_state_and_rng': True, 'same_per_generation_observed_state_and_rng': True,
            'pure_gc_collections': {name: [[run['gc_collections'] for run in record['cases'][key]['runs']]
                                          for record in records] for name, records in runs.items()},
            'optimizer_updates_per_optimizer': observed_samples['current']['optimizer_updates_per_optimizer'],
            'final_population': observed_samples['current']['final_population'],
            'passed_reincarnator_final_position': observed_samples['current']['passed_reincarnator_final_position'],
            'observed_generations': {name: run['generations_observed'] for name, run in observed_samples.items()},
        }
        report['cases'].append(row)
        (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--configuration', choices=CONFIGURATIONS, default='current')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--reverse-cases', action='store_true')
    arguments = parser.parse_args()
    if arguments.blocks < 6 or arguments.blocks % 6 or arguments.samples < 1:
        parser.error('blocks must be a positive multiple of six; samples must be positive')
    worker(arguments) if arguments.worker else parent(arguments)
