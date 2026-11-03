"""A4 golden-one lifecycle contracts; execute only on hosted cloud runners."""
import ast
import functools
import gc
import importlib
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from constants import TITAN_NAMES
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import configure_backend
from population_manager import PopulationManager
import population_manager as population_module


ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'f2073f001af4bc3f03da6d96d5360aa9a2027e0a'


@pytest.fixture
def native(monkeypatch):
    module = configure_backend('cpp')
    monkeypatch.setattr(lifecycle(), 'NATIVE_GOLDEN_ENABLED', True)
    try:
        yield module
    finally:
        configure_backend('python')


def lifecycle():
    try:
        module = importlib.import_module('native_lifecycle')
        assert callable(module.try_update_golden_ones)
        return module
    except (ImportError, AttributeError) as exc:
        pytest.fail(f'Native golden-one lifecycle has not been implemented: {exc}')


@functools.lru_cache(maxsize=1)
def frozen_update():
    source = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'{BASELINE}:population_manager.py'],
        encoding='utf-8')
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'PopulationManager')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'update_golden_ones')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {'random': random, 'TITAN_NAMES': TITAN_NAMES}
    exec(compile(module, '<frozen-A3-golden-ones>', 'exec'), namespace)
    return namespace['update_golden_ones']


def make_world(size=31, cap=20):
    model = TitanToPathModel()
    manager = PopulationManager(set(), {}, np.ones(12), .2, .01, cap, .1,
                                model, 1000., 1., 1000.)
    population = []
    scores = (-3., -0., 0., 5., 5., 12., 12., 1.)
    for index in range(size):
        entity = Pathstrider(f'entity-{index}', np.arange(1., 13.) + index,
                            titan_to_path_model=model)
        value = scores[index % len(scores)]
        entity.score = np.float64(value) if index % 2 else value
        entity.golden_one_tenure = 40 + index
        if index % 4 == 0:
            entity.trait = 'GoldenOne'
            entity.titan_aspect = TITAN_NAMES[index % 12]
            entity.data_modification_unlocked = True
        if index % 7 == 0:
            entity.is_titan_boss = True
        if index % 11 == 1:
            entity.trait = 'Reincarnator'
        if index % 3:
            entity._path_affinities = np.linspace(.01, .12, 12).copy()
        population.append(entity)
        manager.existing_names.add(entity.name)
        manager.name_to_entity_map[entity.name] = entity
    return manager, population


def seed():
    random.seed(731)
    random.gauss(0., 1.)  # Preserve an unrelated cached Gaussian through shuffle.
    np.random.seed(119)
    np.random.normal()
    torch.manual_seed(83)


def rng():
    return pickle.dumps((random.getstate(), np.random.get_state(),
                         torch.get_rng_state().numpy().tobytes()))


def snapshot(manager, population):
    # Raw dictionaries avoid re-running user descriptors when checking their
    # exact callback/failure prefix. A separate mechanism test uses fresh,
    # unmaterialized instances and deliberately does not call this helper.
    values = [{key: value for key, value in vars(entity).items()
               if key != '_titan_to_path_model'} for entity in population]
    order = {id(entity): index for index, entity in enumerate(population)}
    mapping = [(name, order.get(id(entity), -1))
               for name, entity in manager.name_to_entity_map.items()]
    return pickle.dumps((values, sorted(manager.existing_names), mapping))


def close(manager):
    manager.close_native_world()


def outcome(function, manager, population):
    try:
        function(manager, population)
    except Exception as exc:
        return type(exc), str(exc)
    return None


def test_disabled_default_declines_before_backend_metadata_binding_or_input_access(monkeypatch):
    module = lifecycle()
    assert module.NATIVE_GOLDEN_ENABLED is False

    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled golden selection must return before preparation or binding access')

    class Unreadable:
        __getattribute__ = __bool__ = __len__ = __iter__ = forbidden

    for name in ('get_native', '_metadata', '_GoldenSource', '_GoldenView'):
        monkeypatch.setattr(module, name, forbidden)
    seed()
    before = rng()
    assert module.try_update_golden_ones(Unreadable(), Unreadable()) is False
    assert rng() == before


def test_disabled_default_manager_entry_matches_frozen_full_state_and_rng(monkeypatch):
    module = lifecycle()
    assert module.NATIVE_GOLDEN_ENABLED is False
    manager, population = make_world()
    expected_manager, expected = make_world()
    aliases = [(entity, entity.titan_affinities, entity._path_affinities) for entity in population]
    seed()
    frozen_update()(expected_manager, expected)
    expected_state, expected_rng = snapshot(expected_manager, expected), rng()

    def forbidden(*args, **kwargs):
        raise AssertionError('Default golden update must not prepare or bind the native plan')

    original, calls = module.try_update_golden_ones, []

    def observed(actual_manager, actual_population):
        result = original(actual_manager, actual_population)
        calls.append((actual_manager, actual_population, result))
        return result

    for name in ('get_native', '_metadata', '_GoldenSource', '_GoldenView'):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(manager, 'get_native_world', forbidden)
    monkeypatch.setattr(module, 'try_update_golden_ones', observed)
    if hasattr(population_module, 'try_update_golden_ones'):
        monkeypatch.setattr(population_module, 'try_update_golden_ones', observed)
    seed()
    try:
        manager.update_golden_ones(population)
        assert len(calls) == 1 and calls[0][0] is manager and calls[0][1] is population
        assert calls[0][2] is False
        assert snapshot(manager, population) == expected_state and rng() == expected_rng
        assert manager._native_world is None
        for entity, (original_entity, titans, cache) in zip(population, aliases):
            assert entity is original_entity
            assert entity.titan_affinities is titans and entity._path_affinities is cache
    finally:
        close(manager)
        close(expected_manager)


@pytest.mark.parametrize('case,size,cap', [
    ('empty', 0, 12), ('single', 1, 12), ('small', 7, 12),
    ('ties', 31, 20), ('all_excluded', 31, 20),
    ('small_cap', 31, 3), ('negative_cap', 31, -7), ('large_cap', 31, 100),
    ('unknown_trait', 31, 20), ('integer_zero', 31, 20), ('same_names', 31, 20),
])
def test_native_golden_plan_matches_frozen_identity_order_types_aliases_and_rng(native, case, size, cap):
    helper = lifecycle().try_update_golden_ones
    actual_manager, actual = make_world(size, cap)
    expected_manager, expected = make_world(size, cap)
    for manager, population in ((actual_manager, actual), (expected_manager, expected)):
        if case == 'all_excluded':
            for index, entity in enumerate(population):
                if index % 2:
                    entity.trait = 'Reincarnator'
                else:
                    entity.is_titan_boss = True
        elif case == 'unknown_trait':
            population[2].trait = 'ordinary-custom-label'
        elif case == 'integer_zero':
            for entity in population:
                entity.score = 0
        elif case == 'same_names':
            for entity in population:
                entity.name = 'same-name'
    aliases = [(entity, entity.titan_affinities, entity._path_affinities)
               for entity in actual]
    seed()
    initial_rng = rng()
    frozen_update()(expected_manager, expected)
    expected_rng = rng()
    seed()
    try:
        assert helper(actual_manager, actual) is True
        assert rng() == expected_rng
        assert snapshot(actual_manager, actual) == snapshot(expected_manager, expected)
        for entity, (original, titans, cache) in zip(actual, aliases):
            assert entity is original
            assert entity.titan_affinities is titans and entity._path_affinities is cache
        if size:
            assert actual_manager._native_world is not None
            assert actual_manager._native_world.golden_count == 1
            assert expected_rng != initial_rng, 'Nonempty input shuffles even with no candidates'
            assert actual[0].trait == 'Mortal' and actual[0].titan_aspect is None
            assert actual[0].data_modification_unlocked is False
            assert actual[0].golden_one_tenure == 40, 'Clearing an old boss does not reset its tenure'
        else:
            assert expected_rng == initial_rng
    finally:
        close(actual_manager)
        close(expected_manager)


def test_population_manager_routes_standard_golden_update_through_native_helper(native, monkeypatch):
    module = lifecycle()
    original = module.try_update_golden_ones
    calls = []

    def observed(manager, population):
        result = original(manager, population)
        calls.append((manager, population, result))
        return result

    monkeypatch.setattr(module, 'try_update_golden_ones', observed)
    # Both module-qualified and imported-function integration remain acceptable.
    if hasattr(population_module, 'try_update_golden_ones'):
        monkeypatch.setattr(population_module, 'try_update_golden_ones', observed)
    manager, population = make_world()
    expected_manager, expected = make_world()
    seed()
    frozen_update()(expected_manager, expected)
    expected_rng = rng()
    seed()
    try:
        manager.update_golden_ones(population)
        assert len(calls) == 1 and calls[0][0] is manager and calls[0][1] is population
        assert calls[0][2] is True
        assert snapshot(manager, population) == snapshot(expected_manager, expected)
        assert rng() == expected_rng
    finally:
        close(manager)
        close(expected_manager)


def test_golden_plan_reads_only_metadata_and_reuses_a_membership_dirty_world(native, monkeypatch):
    helper = lifecycle().try_update_golden_ones
    manager, population = make_world()
    expected_manager, expected = make_world()
    world = manager.get_native_world()
    world.sync(population)
    population[2].score = expected[2].score = 91.
    world.reconcile(population)
    initial_ids, initial_syncs = world.stable_ids(), world.sync_count

    def forbidden(entity):
        raise AssertionError('Golden selection must not import titan/path vectors')

    monkeypatch.setattr(Pathstrider, 'titan_affinities', property(forbidden), raising=False)
    monkeypatch.setattr(Pathstrider, '_path_affinities', property(forbidden), raising=False)
    seed()
    frozen_update()(expected_manager, expected)
    expected_rng = rng()
    seed()
    try:
        assert helper(manager, population) is True
        assert manager._native_world is world
        assert world.stable_ids() == initial_ids and world.sync_count == initial_syncs
        with pytest.raises(RuntimeError, match='sync'):
            world.snapshot()
        assert snapshot(manager, population) == snapshot(expected_manager, expected)
        assert rng() == expected_rng
    finally:
        close(manager)
        close(expected_manager)


@pytest.mark.skipif(platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 12),
                    reason='This mechanism contract targets CPython 3.12 managed dictionaries')
def test_golden_plan_does_not_materialize_entity_dictionaries(native):
    helper = lifecycle().try_update_golden_ones
    manager, population = make_world()
    control = Pathstrider('control', np.ones(12))

    def has_dict(entity):
        return any(type(value) is dict for value in gc.get_referents(entity))

    assert not has_dict(control)
    vars(control)
    assert has_dict(control), 'GC probe must detect the dictionary side effect'
    assert not any(has_dict(entity) for entity in population)
    seed()
    try:
        assert helper(manager, population) is True
        assert not any(has_dict(entity) for entity in population)
    finally:
        close(manager)


@pytest.mark.parametrize('case', ['nan', 'positive_inf', 'negative_inf', 'large_integer',
                                  'duplicate_entity', 'entity_subclass', 'manager_subclass',
                                  'fractional_cap', 'string_cap', 'incomparable_score',
                                  'missing_score', 'missing_cap'])
def test_unsupported_golden_inputs_preserve_reference_state_rng_and_failure_prefix(native, case):
    helper = lifecycle().try_update_golden_ones
    actual_manager, actual = make_world()
    expected_manager, expected = make_world()
    for manager, population in ((actual_manager, actual), (expected_manager, expected)):
        if case in ('nan', 'positive_inf', 'negative_inf', 'large_integer', 'incomparable_score'):
            population[2].score = {'nan': float('nan'), 'positive_inf': float('inf'),
                                   'negative_inf': -float('inf'), 'large_integer': 2**80 + 1,
                                   'incomparable_score': 'not-a-number'}[case]
        elif case == 'duplicate_entity':
            population.append(population[2])
        elif case == 'entity_subclass':
            class CustomEntity(Pathstrider):
                pass
            population[2].__class__ = CustomEntity
        elif case == 'manager_subclass':
            class CustomManager(PopulationManager):
                pass
            manager.__class__ = CustomManager
        elif case == 'fractional_cap':
            manager.golden_one_cap = 12.5
        elif case == 'string_cap':
            manager.golden_one_cap = 'twenty'
        elif case == 'missing_score':
            del population[2].score
        elif case == 'missing_cap':
            del manager.golden_one_cap
    seed()
    reference_error = outcome(frozen_update(), expected_manager, expected)
    expected_state, expected_rng = snapshot(expected_manager, expected), rng()
    seed()
    before, before_rng = snapshot(actual_manager, actual), rng()
    try:
        assert helper(actual_manager, actual) is False
        assert snapshot(actual_manager, actual) == before and rng() == before_rng
        actual_error = outcome(PopulationManager.update_golden_ones, actual_manager, actual)
        assert actual_error == reference_error
        assert snapshot(actual_manager, actual) == expected_state and rng() == expected_rng
        if case in ('fractional_cap', 'string_cap', 'incomparable_score', 'missing_score', 'missing_cap'):
            assert reference_error is not None
            assert expected_state != before, 'The reference error occurs after identity clearing'
    finally:
        close(actual_manager)
        close(expected_manager)


@pytest.mark.parametrize('field', ['trait', 'score', 'golden_one_cap'])
def test_data_descriptors_fall_back_before_extra_callbacks_or_random_draws(native, monkeypatch, field):
    helper = lifecycle().try_update_golden_ones
    actual_manager, actual = make_world()
    expected_manager, expected = make_world()
    visits = []
    owner = PopulationManager if field == 'golden_one_cap' else Pathstrider

    def read(instance):
        visits.append(('get', field, vars(instance).get('name', 'manager')))
        np.random.random()
        return vars(instance)[field]

    def write(instance, value):
        visits.append(('set', field, vars(instance).get('name', 'manager'), value))
        vars(instance)[field] = value

    monkeypatch.setattr(owner, field, property(read, write), raising=False)
    seed()
    frozen_update()(expected_manager, expected)
    expected_state, expected_rng, expected_visits = snapshot(expected_manager, expected), rng(), list(visits)
    visits.clear()
    seed()
    before, before_rng = snapshot(actual_manager, actual), rng()
    try:
        assert helper(actual_manager, actual) is False
        assert visits == [] and rng() == before_rng
        assert snapshot(actual_manager, actual) == before
        actual_manager.update_golden_ones(actual)
        assert visits == expected_visits
        assert snapshot(actual_manager, actual) == expected_state and rng() == expected_rng
    finally:
        close(actual_manager)
        close(expected_manager)


def test_python_backend_keeps_golden_reference_path_without_resident_allocation(monkeypatch):
    module = lifecycle()
    monkeypatch.setattr(module, 'NATIVE_GOLDEN_ENABLED', True)
    helper = module.try_update_golden_ones
    configure_backend('python')
    manager, population = make_world()
    expected_manager, expected = make_world()
    seed()
    frozen_update()(expected_manager, expected)
    expected_state, expected_rng = snapshot(expected_manager, expected), rng()
    seed()
    before, before_rng = snapshot(manager, population), rng()
    try:
        assert helper(manager, population) is False
        assert manager._native_world is None
        assert snapshot(manager, population) == before and rng() == before_rng
        manager.update_golden_ones(population)
        assert snapshot(manager, population) == expected_state and rng() == expected_rng
        assert manager._native_world is None
    finally:
        close(manager)
        close(expected_manager)
        configure_backend('python')


def test_first_helper_import_with_custom_shuffle_keeps_original_failure_prefix(native, monkeypatch):
    manager, population = make_world()
    expected_manager, expected = make_world()
    original_shuffle = random.shuffle
    visits = []

    def custom_shuffle(values):
        visits.append(tuple(values))
        original_shuffle(values)
        raise RuntimeError('custom shuffle failed after its original draws')

    monkeypatch.setattr(random, 'shuffle', custom_shuffle)
    # Reproduce a fresh adapter import after a pre-existing module-level hook.
    # The helper must reject a plain function before reading MethodType fields.
    monkeypatch.delitem(sys.modules, 'native_lifecycle')
    module = lifecycle()
    monkeypatch.setattr(module, 'NATIVE_GOLDEN_ENABLED', True)
    helper = module.try_update_golden_ones
    seed()
    expected_error = outcome(frozen_update(), expected_manager, expected)
    expected_state, expected_rng, expected_visits = snapshot(expected_manager, expected), rng(), list(visits)
    visits.clear()
    seed()
    before, before_rng = snapshot(manager, population), rng()
    try:
        assert helper(manager, population) is False
        assert visits == [] and rng() == before_rng
        assert snapshot(manager, population) == before
        assert outcome(PopulationManager.update_golden_ones, manager, population) == expected_error
        assert expected_error == (RuntimeError, 'custom shuffle failed after its original draws')
        assert snapshot(manager, population) == expected_state
        assert rng() == expected_rng and visits == expected_visits
    finally:
        close(manager)
        close(expected_manager)
