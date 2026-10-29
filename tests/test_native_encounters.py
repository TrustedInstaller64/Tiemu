"""Remote-only contracts for an interpreter-free ordered encounter engine.

The oracle loads the frozen A1 loop and handler/manager definitions from Git.
It deliberately retains fixed old buckets, lazy-cache timing and RNG consumption.
Do not run these tests on the reserved development Mac.
"""
import ast
import copy
import functools
import importlib
import logging
from pathlib import Path
import pickle
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.xfail(raises=pytest.fail.Exception, strict=True,
                              reason='Remote RED: the standalone encounter bridge is absent')

from constants import PATH_NAMES, TITAN_NAMES
from entities import Pathstrider
from interaction_handler import InteractionHandler
from models import TitanToPathModel
from native_backend import configure_backend
from parliament_manager import ParliamentManager
from population_manager import PopulationManager


BASELINE = '4a2311dfff44e5b92829dfdb84f735f37c171701'
ROOT = Path(__file__).resolve().parents[1]


def candidate():
    try:
        return importlib.import_module('native_encounters')
    except ImportError as exc:
        pytest.fail(f'Pure C ABI encounter bridge has not been implemented: {exc}')


@pytest.fixture
def native():
    module = configure_backend('cpp')
    try:
        yield module
    finally:
        configure_backend('python')


def source_at_baseline(filename):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{BASELINE}:{filename}'], text=True)


@functools.lru_cache(maxsize=1)
def frozen_oracle():
    classes = {}
    for filename, name in [('interaction_handler.py', 'InteractionHandler'),
                           ('population_manager.py', 'PopulationManager'),
                           ('parliament_manager.py', 'ParliamentManager')]:
        namespace = {'__name__': f'_encounter_oracle_{name}'}
        exec(compile(source_at_baseline(filename), f'{BASELINE}/{filename}', 'exec'), namespace)
        classes[name] = namespace[name]
    tree = ast.parse(source_at_baseline('simulation.py'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AeonEvolution')
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_run_one_generation')
    # The exact frozen block starts at the update set, includes update_bins and
    # the tracked encounter loop, and ends before global events and final scoring.
    start = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'entities_to_update'
                         for target in node.targets))
    block = []
    for node in function.body[start:]:
        block.append(copy.deepcopy(node))
        if isinstance(node, ast.With):
            break
    tracked = block[-1]
    assert isinstance(tracked, ast.With)
    loop = next(node for node in tracked.body if isinstance(node, ast.For))
    loop.body[0:0] = ast.parse('culled_entity = None\n__applied = False').body
    application = next(node for node in loop.body if isinstance(node, ast.If)
                       and 'entity1 not in culled_this_gen' in ast.unparse(node.test))
    application.body[0:0] = ast.parse('__applied = True').body
    loop.body.extend(ast.parse(
        '__trace.append(__record(_, entity1, entity2, culled_entity, __applied))').body)
    block.insert(0, ast.parse('__trace = []').body[0])
    block.append(ast.parse('return culled_this_gen, entities_to_update, __trace').body[0])
    function.name = 'reference_encounters'
    function.body = block
    function.decorator_list = []
    function.args.args.append(ast.arg(arg='__record'))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    from native_backend import get_native
    namespace = {'get_native': get_native}
    exec(compile(module, '<frozen-A1-encounter-loop>', 'exec'), namespace)
    return classes, namespace['reference_encounters']


def make_world(size=16, relation='SYNERGY', kind='mortal', frozen=False):
    classes = frozen_oracle()[0] if frozen else {
        'PopulationManager': PopulationManager, 'InteractionHandler': InteractionHandler,
        'ParliamentManager': ParliamentManager}
    model = TitanToPathModel()
    manager = classes['PopulationManager'](set(), {}, np.ones(12), 0.05, 0.01, 12, 0.1,
                                          model, 100.0, 1.0, 1000.0)
    manager.cosmic_tide_vector = np.linspace(-0.002, 0.003, 12)
    local = np.random.RandomState(1701)
    population = [Pathstrider(f'entity-{i}', local.uniform(0.2, 4.0, 12), titan_to_path_model=model)
                  for i in range(size)]
    for i, entity in enumerate(population):
        entity.score = 3.0 + (i % 7) * 0.3
        entity.activity, entity.stability, entity.heroic_tendency = 1.0+i, 2.0+i, 1.0+i
        if i % 2:
            entity._path_affinities = np.roll(np.arange(1., 13.)/78., i)
    relations = np.full((12, 12), relation, dtype=object)
    modifiers = np.arange(144, dtype=np.float64).reshape(12, 12)/200.0 + 0.5
    handler = classes['InteractionHandler'](relations, modifiers, 0.5, 0.0, size, model, manager)
    parliament = classes['ParliamentManager']()
    parliament.seats = {name: i+1 for i, name in enumerate(PATH_NAMES)}
    reincarnator = None
    if kind in ('reincarnator', 'destroy', 'golden'):
        reincarnator = population[0]
        reincarnator.trait = 'Reincarnator'
        population[0]._path_affinities = np.eye(12)[0].copy()
        population[1]._path_affinities = np.eye(12)[0 if kind == 'destroy' else 1].copy()
        if kind == 'golden':
            population[1].trait = 'GoldenOne'
    if kind in ('battle', 'defeat'):
        population[0].trait = 'GoldenOne'
        population[0].titan_aspect = TITAN_NAMES[0]
        population[1].is_titan_boss = True
        population[1].titan_aspect = TITAN_NAMES[0]
        population[1].hp = 0.001 if kind == 'defeat' else 10000.0
    return SimpleNamespace(population=population, population_manager=manager,
                           parliament_manager=parliament, interaction_handler=handler,
                           reincarnator=reincarnator, cosmic_zeitgeist=np.ones(12)/12)


def rng_state():
    return pickle.dumps((random.getstate(), np.random.get_state()))


def trace_recorder(population):
    indices = {entity: i for i, entity in enumerate(population)}
    def record(attempt, first, second, culled, applied):
        numpy_state = np.random.get_state()
        return {'attempt': attempt, 'first': indices.get(first, -1),
                'second': indices.get(second, -1), 'culled': indices.get(culled, -1),
                'applied': bool(applied),
                'numpy_state': [int(x) for x in numpy_state[1]] + [numpy_state[2]],
                'python_state': list(random.getstate()[1])}
    return record


def snapshot(world):
    entities = []
    for entity in world.population:
        entities.append({**entity.to_dict(), 'activity': entity.activity, 'stability': entity.stability,
                         'heroic_tendency': entity.heroic_tendency,
                         'cache': None if entity._path_affinities is None else entity._path_affinities.copy()})
    return entities


def assert_same_entities(actual, expected):
    assert actual.parliament_manager._seat_cache_key == expected.parliament_manager._seat_cache_key
    left_seats = actual.parliament_manager._normalized_seat_vector
    right_seats = expected.parliament_manager._normalized_seat_vector
    if left_seats is None or right_seats is None:
        assert left_seats is None and right_seats is None
    else:
        np.testing.assert_array_equal(left_seats, right_seats)
    for left, right in zip(snapshot(actual), snapshot(expected)):
        for key in left:
            if key == 'cache' and (left[key] is None or right[key] is None):
                assert left[key] is None and right[key] is None
            elif key in ('cache', 'titan_affinities'):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                assert left[key] == right[key], key


def ids(population, entities):
    return {i for i, entity in enumerate(population) if entity in entities}


def close_index(world):
    index = world.interaction_handler._native_opponents
    if index is not None and hasattr(index, 'close'):
        index.close()


@pytest.mark.parametrize('relation', ['NEUTRAL', 'CLASH', 'SYNERGY', 'REPULSION', 'MENTORSHIP'])
@pytest.mark.parametrize('seed', [7, 731])
def test_ordered_encounter_loop_matches_frozen_a1_at_every_rng_boundary(native, relation, seed):
    bridge = candidate()
    actual, expected = make_world(relation=relation), make_world(relation=relation, frozen=True)
    actual_aliases = [(entity.titan_affinities, entity._path_affinities) for entity in actual.population]
    expected_aliases = [(entity.titan_affinities, entity._path_affinities) for entity in expected.population]
    random.seed(seed)
    np.random.seed(seed)
    expected_culled, expected_updated, expected_trace = frozen_oracle()[1](
        expected, trace_recorder(expected.population))
    expected_rng = rng_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        result = bridge.try_run_encounters(actual, trace=True)
        assert result is not None, 'Ordinary built-in entities must use the pure C++ loop'
        assert result.trace == expected_trace
        assert rng_state() == expected_rng
        assert ids(actual.population, result.culled) == ids(expected.population, expected_culled)
        assert ids(actual.population, result.updated) == ids(expected.population, expected_updated)
        assert_same_entities(actual, expected)
        for entity, reference, old, old_reference in zip(
                actual.population, expected.population, actual_aliases, expected_aliases):
            np.testing.assert_array_equal(old[0], old_reference[0])
            assert (entity.titan_affinities is old[0]) == (reference.titan_affinities is old_reference[0])
            assert (entity._path_affinities is old[1]) == (reference._path_affinities is old_reference[1])
        assert actual.interaction_handler._native_opponents.alive_count == expected.interaction_handler._native_opponents.alive_count
    finally:
        close_index(actual)
        close_index(expected)


@pytest.mark.parametrize('kind,seed', [('reincarnator', 731), ('golden', 731), ('destroy', 731),
                                      ('battle', 15), ('defeat', 731)])
def test_special_entities_logs_old_aliases_and_cached_paths_are_preserved(native, kind, seed, caplog):
    bridge = candidate()
    actual, expected = make_world(size=2, kind=kind), make_world(size=2, kind=kind, frozen=True)
    old_actual = [entity.titan_affinities for entity in actual.population]
    old_expected = [entity.titan_affinities for entity in expected.population]
    caplog.set_level(logging.INFO, logger='OmphalosLogger')
    random.seed(seed)
    np.random.seed(seed)
    expected_culled, expected_updated, expected_trace = frozen_oracle()[1](
        expected, trace_recorder(expected.population))
    expected_logs = [record.getMessage() for record in caplog.records]
    expected_rng = rng_state()
    caplog.clear()
    random.seed(seed)
    np.random.seed(seed)
    try:
        result = bridge.try_run_encounters(actual, trace=True)
        assert result is not None
        assert result.trace == expected_trace
        assert rng_state() == expected_rng
        assert [record.getMessage() for record in caplog.records] == expected_logs
        assert ids(actual.population, result.culled) == ids(expected.population, expected_culled)
        assert ids(actual.population, result.updated) == ids(expected.population, expected_updated)
        assert_same_entities(actual, expected)
        for left, right in zip(old_actual, old_expected):
            np.testing.assert_array_equal(left, right)
        if kind == 'battle':
            assert any('灰飞烟灭' in line for line in expected_logs)
        if kind == 'destroy':
            assert any('无条件的毁灭' in line for line in expected_logs)
    finally:
        close_index(actual)
        close_index(expected)


def test_stale_bucket_draws_and_culling_keep_numpy_rng_consumption(native):
    bridge = candidate()
    actual, expected = make_world(size=24), make_world(size=24, frozen=True)
    for world in (actual, expected):
        world.interaction_handler.culling_strength = 100.0
        for i, entity in enumerate(world.population):
            entity.score = 2.0 + i/10.0
    random.seed(73)
    np.random.seed(73)
    expected_culled, expected_updated, expected_trace = frozen_oracle()[1](
        expected, trace_recorder(expected.population))
    expected_rng = rng_state()
    random.seed(73)
    np.random.seed(73)
    try:
        result = bridge.try_run_encounters(actual, trace=True)
        assert result is not None
        assert result.trace == expected_trace
        assert rng_state() == expected_rng
        assert any(not row['applied'] for row in expected_trace), 'Fixture must select a dead old-bin opponent'
        assert_same_entities(actual, expected)
    finally:
        close_index(actual)
        close_index(expected)


@pytest.mark.parametrize('unsafe', ['purity_zero', 'purity_negative', 'shared_array', 'view',
                                   'custom_handler', 'duplicate', 'nonfinite', 'different_matrix'])
def test_ineligible_input_returns_none_without_mutating_state_or_rng(native, unsafe):
    bridge = candidate()
    world = make_world()
    if unsafe == 'purity_zero':
        world.population_manager.purity_factor = 0.0
    elif unsafe == 'purity_negative':
        world.population_manager.purity_factor = -0.1
    elif unsafe == 'shared_array':
        world.population[1].titan_affinities = world.population[0].titan_affinities
    elif unsafe == 'view':
        world.population[0].titan_affinities = world.population[0].titan_affinities[::-1]
    elif unsafe == 'custom_handler':
        class CustomHandler(InteractionHandler):
            pass
        world.interaction_handler.__class__ = CustomHandler
    elif unsafe == 'duplicate':
        world.population[1] = world.population[0]
    elif unsafe == 'nonfinite':
        world.population[0].titan_affinities[0] = np.nan
    elif unsafe == 'different_matrix':
        world.population[0]._titan_to_path_model = TitanToPathModel()
    before = pickle.dumps(snapshot(world))
    before_rng = rng_state()
    assert world.interaction_handler._native_opponents is None
    assert bridge.try_run_encounters(world, trace=True) is None
    assert pickle.dumps(snapshot(world)) == before
    assert rng_state() == before_rng
    assert world.interaction_handler._native_opponents is None


def test_production_call_omits_per_encounter_trace_and_preserves_cached_gaussians(native):
    bridge = candidate()
    world = make_world()
    random.seed(731)
    np.random.seed(731)
    random.gauss(0, 1)
    np.random.normal()
    python_gauss = random.getstate()[2]
    numpy_gauss = np.random.get_state()[3:]
    try:
        result = bridge.try_run_encounters(world)
        assert result is not None
        assert result.trace == []
        assert random.getstate()[2] == python_gauss
        assert np.random.get_state()[3:] == numpy_gauss
    finally:
        close_index(world)
