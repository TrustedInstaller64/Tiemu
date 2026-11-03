"""A3 birth migration contracts; run only on the hosted cloud runners."""
import ast
import functools
import importlib
from pathlib import Path
import pickle
import random
import subprocess

import numpy as np
import pytest

from constants import GREEK_ROOTS
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import configure_backend
from population_manager import PopulationManager


BASELINE = '26c1c2624cd85529e801754093bf061e0a316ec4'
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native():
    module = configure_backend('cpp')
    try:
        yield module
    finally:
        configure_backend('python')


def resident_type():
    try:
        return importlib.import_module('native_world').ResidentWorld
    except (ImportError, AttributeError) as exc:
        pytest.fail(f'Resident native birth pipeline has not been implemented: {exc}')


@functools.lru_cache(maxsize=1)
def frozen_births():
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                      f'{BASELINE}:population_manager.py'], encoding='utf-8')
    tree = ast.parse(source)
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef)
               and item.name == 'PopulationManager')
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef)
                  and item.name == '_add_new_entities')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {'np': np, 'Pathstrider': Pathstrider}
    exec(compile(module, '<frozen-A2-births>', 'exec'), namespace)
    return namespace['_add_new_entities']


def fixture_world():
    model = TitanToPathModel()
    blueprint = np.linspace(-.2, 1.5, 12)
    manager = PopulationManager(set(), {}, blueprint, .25, .01, 12, .1,
                                model, 4., 1., 1000.)
    manager.cosmic_tide_vector = np.linspace(-.03, .01, 12)
    population = [Pathstrider(f'prior-{i}', np.arange(1., 13.) + i,
                             titan_to_path_model=model) for i in range(3)]
    for entity in population:
        manager.existing_names.add(entity.name)
        manager.name_to_entity_map[entity.name] = entity
    # Force the first ordinary generated name to collide. Names and Gaussian
    # draws must retain their separate streams and exact continuation state.
    naming = random.Random(731)
    roots = naming.sample(GREEK_ROOTS, 2)
    manager.existing_names.add(f'{roots[0]}{roots[1]}-{naming.randint(1000, 9999)}')
    return manager, population


def capture_rng():
    return pickle.dumps((random.getstate(), np.random.get_state()))


def seed():
    random.seed(731)
    np.random.seed(731)
    np.random.normal()  # Populate RandomState's cached Gaussian.


def assert_same(manager, entities, expected_manager, expected):
    assert [p.name for p in entities] == [p.name for p in expected]
    assert manager.existing_names == expected_manager.existing_names
    assert list(manager.name_to_entity_map) == list(expected_manager.name_to_entity_map)
    for actual, reference in zip(entities, expected):
        assert actual.to_dict() == reference.to_dict()
        for name in ('score', 'activity', 'stability', 'heroic_tendency'):
            assert type(getattr(actual, name)) is type(getattr(reference, name))
            assert getattr(actual, name) == getattr(reference, name)
        if reference._path_affinities is None:
            assert actual._path_affinities is None
        else:
            np.testing.assert_array_equal(actual._path_affinities, reference._path_affinities)
        assert manager.name_to_entity_map[actual.name] is actual


@pytest.mark.parametrize('count', [1, 7, 64])
@pytest.mark.parametrize('legacy', [False, True])
def test_native_births_match_frozen_order_rng_types_and_lazy_caches(native, monkeypatch, count, legacy):
    cls = resident_type()
    calls = []
    original = cls.births

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, 'births', counted)
    actual_manager, actual = fixture_world()
    expected_manager, expected = fixture_world()
    modifier = np.linspace(-.02, .01, 12) if legacy else None
    seed()
    frozen_births()(expected_manager, expected, count, expected_manager.base_titan_affinities,
                    np.ones(12), modifier)
    expected_rng = capture_rng()
    seed()
    try:
        actual_manager._add_new_entities(actual, count, actual_manager.base_titan_affinities,
                                         np.ones(12), modifier)
        assert capture_rng() == expected_rng
        assert_same(actual_manager, actual, expected_manager, expected)
        assert len(calls) == 1, 'Normal births must execute one whole native numerical segment'
        assert calls[0] is actual_manager._native_world
        assert len(calls[0].stable_ids()) == len(actual)
        # Births reconcile membership without redundantly validating/uploading
        # every old row. Numeric consumers must explicitly synchronize first.
        with pytest.raises(RuntimeError, match='sync'):
            calls[0].snapshot()
        calls[0].sync(actual)
        np.testing.assert_array_equal(calls[0].snapshot()['titans'],
                                      np.stack([entity.titan_affinities for entity in actual]))
        assert all(entity.titan_affinities.flags.owndata for entity in actual)
    finally:
        world = getattr(actual_manager, '_native_world', None)
        if world is not None:
            world.close()


@pytest.mark.parametrize('case', ['custom_name', 'nonpositive_purity', 'custom_constructor',
                                  'large_finite', 'raise_on_overflow', 'custom_new',
                                  'model_getattribute', 'model_feedback_property',
                                  'manager_data_property', 'random_class_randbelow',
                                  'random_class_getrandbits'])
def test_unsupported_birth_path_preserves_ordered_reference_side_effects(native, monkeypatch, case):
    cls = resident_type()
    native_calls = []
    original = cls.births

    def counted(self, *args, **kwargs):
        native_calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, 'births', counted)
    actual_manager, actual = fixture_world()
    expected_manager, expected = fixture_world()
    modifier = None
    active_manager = [expected_manager]
    if case == 'custom_name':
        for manager in (actual_manager, expected_manager):
            original_name = manager.generate_unique_name
            def name(original_name=original_name):
                np.random.random()  # Cross-stream user callback must stay ordered.
                return original_name()
            manager.generate_unique_name = name
    elif case == 'nonpositive_purity':
        actual_manager.purity_factor = expected_manager.purity_factor = 0.0
    elif case == 'custom_constructor':
        constructor = Pathstrider.__init__
        def initialize(self, *args, **kwargs):
            random.random()
            constructor(self, *args, **kwargs)
        monkeypatch.setattr(Pathstrider, '__init__', initialize)
    elif case == 'large_finite':
        actual_manager.base_titan_affinities[:] = 1e150
        expected_manager.base_titan_affinities[:] = 1e150
    elif case == 'custom_new':
        def create(entity_cls, *args, **kwargs):
            np.random.random()
            return object.__new__(entity_cls)
        monkeypatch.setattr(Pathstrider, '__new__', staticmethod(create))
    elif case == 'model_getattribute':
        modifier = np.ones(12) * .01
        def read(model, name):
            if name == 'path_to_titan_feedback_matrix':
                np.random.random()
            return object.__getattribute__(model, name)
        monkeypatch.setattr(TitanToPathModel, '__getattribute__', read)
    elif case == 'model_feedback_property':
        modifier = np.ones(12) * .01
        def feedback(model):
            np.random.random()
            return model.__dict__['path_to_titan_feedback_matrix']
        monkeypatch.setattr(TitanToPathModel, 'path_to_titan_feedback_matrix',
                            property(feedback), raising=False)
    elif case == 'manager_data_property':
        def mutation(manager):
            np.random.random()
            return manager.__dict__['mutation_rate']
        monkeypatch.setattr(PopulationManager, 'mutation_rate', property(mutation), raising=False)
    elif case in ('random_class_randbelow', 'random_class_getrandbits'):
        name = '_randbelow' if case == 'random_class_randbelow' else 'getrandbits'
        original_random = getattr(random.Random, name)
        def generate(owner, *args, **kwargs):
            # Name generation can legally call through a class customization.
            # Delaying every blueprint read until all names exist changes the
            # first newborn if such a callback is accepted by the batch path.
            active_manager[0].base_titan_affinities[:] += .01
            return original_random(owner, *args, **kwargs)
        monkeypatch.setattr(random.Random, name, generate)
    error_settings = np.seterr(over='raise') if case == 'raise_on_overflow' else None
    try:
        seed()
        frozen_births()(expected_manager, expected, 5, expected_manager.base_titan_affinities,
                        None, modifier)
        expected_rng = capture_rng()
        active_manager[0] = actual_manager
        seed()
        actual_manager._add_new_entities(actual, 5, actual_manager.base_titan_affinities,
                                         None, modifier)
        assert capture_rng() == expected_rng
        assert_same(actual_manager, actual, expected_manager, expected)
        assert native_calls == []
    finally:
        if error_settings is not None:
            np.seterr(**error_settings)
        world = getattr(actual_manager, '_native_world', None)
        if world is not None:
            world.close()


def test_birth_outputs_survive_later_native_calls_and_close(native):
    world = resident_type()()
    try:
        blueprint = np.linspace(-1., 2., 12)
        noise = np.arange(48, dtype=np.float64).reshape(4, 12) / 100.
        tide = np.linspace(-.1, .1, 12)
        first = world.births(blueprint, noise, None, tide, .01, 4.)
        saved = first.copy()
        world.births(blueprint * 2., noise, np.ones(12), tide, .01, 4.)
        world.close()
        np.testing.assert_array_equal(first, saved)
        assert first.flags.owndata
    finally:
        world.close()


@pytest.mark.parametrize('case', ['negative_mutation', 'invalid_modifier', 'replaced_rng'])
def test_birth_fallback_preserves_exception_prefix_without_extra_names_or_rng(native, monkeypatch, case):
    cls = resident_type()
    calls = []
    original = cls.births

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, 'births', counted)
    actual_manager, actual = fixture_world()
    expected_manager, expected = fixture_world()
    modifier = None
    if case == 'negative_mutation':
        actual_manager.mutation_rate = expected_manager.mutation_rate = -0.1
    elif case == 'invalid_modifier':
        modifier = np.ones(11)
    else:
        original_normal = np.random.normal
        def normal(*args, **kwargs):
            random.random()
            return original_normal(*args, **kwargs)
        monkeypatch.setattr(np.random, 'normal', normal)
    seed()
    if case == 'replaced_rng':
        frozen_births()(expected_manager, expected, 5, expected_manager.base_titan_affinities,
                        None, modifier)
    else:
        with pytest.raises(ValueError):
            frozen_births()(expected_manager, expected, 5, expected_manager.base_titan_affinities,
                            None, modifier)
    expected_rng = capture_rng()
    seed()
    try:
        if case == 'replaced_rng':
            actual_manager._add_new_entities(actual, 5, actual_manager.base_titan_affinities,
                                             None, modifier)
        else:
            with pytest.raises(ValueError):
                actual_manager._add_new_entities(actual, 5, actual_manager.base_titan_affinities,
                                                 None, modifier)
        assert capture_rng() == expected_rng
        assert_same(actual_manager, actual, expected_manager, expected)
        assert calls == []
    finally:
        world = getattr(actual_manager, '_native_world', None)
        if world is not None:
            world.close()
