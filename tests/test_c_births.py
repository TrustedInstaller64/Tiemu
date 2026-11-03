"""C2 newborn contracts. All execution is restricted to hosted CI.

The RED gate permits only the genuinely missing batch method to xfail. The
reference below preserves the pre-C2 birth loop; it intentionally keeps one
name call followed by one NumPy normal call for every entity.
"""
import gc
import importlib
import pickle
import random
import weakref

import numpy as np
import pytest

from config import config
from constants import GREEK_ROOTS
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import configure_backend
from population_manager import PopulationManager


_native_module = importlib.import_module("_tiemu_ffi")
pytestmark = pytest.mark.xfail(
    not hasattr(_native_module.BatchEvaluator, "newborn_affinities"),
    raises=AttributeError, strict=True,
    reason="C2 RED: standalone batch newborn API is not implemented")


@pytest.fixture
def native(monkeypatch):
    module = configure_backend("cpp")
    try:
        getattr(module.BatchEvaluator, "newborn_affinities")
        monkeypatch.setitem(config["runtime"], "experimental_c_newborns", True)
        yield module
    finally:
        configure_backend("python")


def numerical_inputs(rows=37):
    rng = np.random.default_rng(731)
    return (np.linspace(-1., 3., 12), rng.normal(0., .3, (rows, 12)),
            np.linspace(-.02, .03, 12), .03, np.linspace(-.1, .2, 12), 2.)


def scalar_newborns(native, blueprint, noise, modifier, factor, tide, limit):
    output = np.empty(noise.shape)
    for index, row in enumerate(noise):
        values = np.array(blueprint + row).clip(min=0)
        if modifier is not None:
            values += modifier * 50.
        output[index] = native.normalize_pipeline(values, factor, tide, limit)
    return output


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("rows", [0, 1, 37])
@pytest.mark.parametrize("legacy", [False, True])
def test_batch_births_keep_clip_legacy_purity_tide_normalization_order(native, workers, rows, legacy):
    args = list(numerical_inputs(rows))
    if not legacy:
        args[2] = None
    before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    evaluator = native.BatchEvaluator(workers, 1)
    try:
        actual = evaluator.newborn_affinities(*args)
    finally:
        evaluator.close()
    np.testing.assert_array_equal(actual, scalar_newborns(native, *args))
    assert actual.shape == (rows, 12)
    for value, previous in zip(args, before):
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, previous)


@pytest.mark.parametrize("case", ["zero", "tied_dominant", "skip_purity", "nan", "infinity"])
def test_birth_kernel_edge_values_match_existing_scalar_pipeline(native, case):
    args = list(numerical_inputs(5))
    if case == "zero":
        args = [np.zeros(12), np.zeros((5, 12)), None, .03, np.zeros(12), 2.]
    elif case == "tied_dominant":
        args[0], args[1], args[2] = np.ones(12), np.zeros((5, 12)), None
    elif case == "skip_purity":
        args[3] = 0.
    elif case == "nan":
        args[1][0, 0] = np.nan
    else:
        args[1][0, 0] = np.inf
    evaluator = native.BatchEvaluator(4, 1)
    try:
        with np.errstate(all="ignore"):
            actual = evaluator.newborn_affinities(*args)
            expected = scalar_newborns(native, *args)
    finally:
        evaluator.close()
    np.testing.assert_array_equal(actual, expected)


def test_parallel_birth_rows_and_reuse_are_bitwise_identical(native):
    args = numerical_inputs(129)
    one, four = native.BatchEvaluator(1, 1), native.BatchEvaluator(4, 1)
    try:
        expected = one.newborn_affinities(*args)
        actual = four.newborn_affinities(*args)
        four.newborn_affinities(*numerical_inputs(2))
    finally:
        one.close()
        four.close()
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(RuntimeError, match="closed"):
        four.newborn_affinities(*args)


@pytest.mark.parametrize("index,bad", [(0, np.ones((1, 12))), (1, np.ones(12)),
                                      (1, np.ones((37, 11))), (2, np.ones(11)),
                                      (4, np.ones(11))])
def test_birth_shapes_are_rejected_before_native_access(native, index, bad):
    args = list(numerical_inputs())
    args[index] = bad
    evaluator = native.BatchEvaluator(1, 1)
    try:
        with pytest.raises(ValueError):
            evaluator.newborn_affinities(*args)
    finally:
        evaluator.close()


def test_birth_c_abi_validates_output_capacity(native, monkeypatch):
    call = native._batch_newborn_affinities
    def truncate(*args):
        args = list(args)
        args[-1] = 0
        return call(*args)
    monkeypatch.setattr(native, "_batch_newborn_affinities", truncate)
    evaluator = native.BatchEvaluator(1, 1)
    try:
        with pytest.raises(ValueError):
            evaluator.newborn_affinities(*numerical_inputs())
    finally:
        evaluator.close()


def test_birth_converted_buffers_remain_owned_through_parallel_completion(native, monkeypatch):
    refs = []
    convert, call = native._array, native._batch_newborn_affinities
    def observe(*args, **kwargs):
        result = convert(*args, **kwargs)
        refs.append(weakref.ref(result))
        return result
    def collect(*args):
        gc.collect()
        assert refs and all(ref() is not None for ref in refs)
        return call(*args)
    monkeypatch.setattr(native, "_array", observe)
    monkeypatch.setattr(native, "_batch_newborn_affinities", collect)
    args = [a.tolist() if isinstance(a, np.ndarray) else a for a in numerical_inputs()]
    evaluator = native.BatchEvaluator(4, 1)
    try:
        actual = evaluator.newborn_affinities(*args)
    finally:
        evaluator.close()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert actual.shape == (37, 12)


def make_manager(factory=None, cls=PopulationManager):
    model = TitanToPathModel()
    manager = cls(set(), {}, np.linspace(-1., 3., 12), .3, .03, 12, .1,
                  model, 2., 1., 10., native_evaluator_factory=factory)
    manager.cosmic_tide_vector = np.linspace(-.1, .2, 12)
    return manager


def legacy_birth_loop(manager, count, population, blueprint, legacy=None, initial=False):
    # Literal pre-C2 order, including the global-distribution cache side effect.
    distribution = None if initial else manager.get_global_path_distribution(population)
    for _ in range(count):
        name = manager.generate_unique_name()
        affinities = blueprint + np.random.normal(0, manager.mutation_rate,
                                                  manager.base_titan_affinities.shape)
        entity = Pathstrider(name, affinities, titan_to_path_model=manager.titan_to_path_model)
        if legacy is not None:
            modifier = np.dot(legacy, manager.titan_to_path_model.path_to_titan_feedback_matrix)
            entity.titan_affinities += modifier * 50.
        manager.recalculate_and_normalize_entity(entity, distribution, None)
        population.append(entity)
        manager.name_to_entity_map[name] = entity


def rng_start():
    random.seed(731)
    np.random.seed(731)
    random.gauss(0., 1.)
    np.random.normal()  # Populate the legacy normal generator's cached value.
    return random.getstate(), np.random.get_state()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])


def snapshot_rng():
    return pickle.dumps((random.getstate(), np.random.get_state()))


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("workers", [1, 4])
def test_application_preserves_names_collision_rng_fields_and_array_ownership(native, monkeypatch, initial, workers):
    evaluator = native.BatchEvaluator(workers, 1)
    actual, expected = make_manager(lambda: evaluator), make_manager()
    state = rng_start()
    collision_rng = random.Random()
    collision_rng.setstate(state[0])
    roots = collision_rng.sample(GREEK_ROOTS, 2)
    collision = f"{roots[0]}{roots[1]}-{collision_rng.randint(1000, 9999)}"
    actual.existing_names.add(collision)
    expected.existing_names.add(collision)
    actual_population, expected_population = [], []
    legacy = None if initial else np.linspace(-.04, .05, 12)
    calls = []
    original = native.BatchEvaluator.newborn_affinities
    def observe(self, *args):
        calls.append(1)
        return original(self, *args)
    monkeypatch.setattr(native.BatchEvaluator, "newborn_affinities", observe)
    try:
        legacy_birth_loop(expected, 19, expected_population, expected.base_titan_affinities, legacy, initial)
        expected_rng = snapshot_rng()
        restore_rng(state)
        if initial:
            actual.create_initial_population(19, actual_population, None)
        else:
            actual._add_new_entities(actual_population, 19, actual.base_titan_affinities, None, legacy)
    finally:
        evaluator.close()
    assert calls == [1], "The newborn batch must actually execute"
    assert snapshot_rng() == expected_rng
    assert actual.existing_names == expected.existing_names
    assert list(actual.name_to_entity_map) == list(expected.name_to_entity_map)
    assert [e.name for e in actual_population] == [e.name for e in expected_population]
    for index, (current, previous) in enumerate(zip(actual_population, expected_population)):
        assert current.__dict__.keys() == previous.__dict__.keys()
        for key, value in previous.__dict__.items():
            if key == '_titan_to_path_model':
                assert current._titan_to_path_model is actual.titan_to_path_model
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(current.__dict__[key], value)
            else:
                assert current.__dict__[key] == value, key
        assert type(current.score) is int and current.score == 0
        assert current._path_affinities is None
        assert current.titan_affinities.flags.owndata
        assert actual.name_to_entity_map[current.name] is current
        for other in actual_population[:index]:
            assert not np.shares_memory(current.titan_affinities, other.titan_affinities)
            assert current.held_fire_seeds is not other.held_fire_seeds


@pytest.mark.parametrize("case", ["flag_off", "no_factory", "custom_normalizer", "custom_rng"])
def test_birth_unsupported_or_disabled_routes_retain_scalar_calls(native, monkeypatch, case):
    evaluator = native.BatchEvaluator(1, 1)
    manager = make_manager(None if case == "no_factory" else lambda: evaluator)
    calls = []
    def forbidden(*args):
        raise AssertionError("Unsupported birth route must stay scalar")
    monkeypatch.setattr(native.BatchEvaluator, "newborn_affinities", forbidden)
    if case == "flag_off":
        monkeypatch.setitem(config["runtime"], "experimental_c_newborns", False)
    elif case == "custom_normalizer":
        original = manager.recalculate_and_normalize_entity
        def normalize(*args):
            calls.append(1)
            return original(*args)
        manager.recalculate_and_normalize_entity = normalize
    elif case == "custom_rng":
        original = np.random.normal
        def normal(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(np.random, "normal", normal)
    population = []
    try:
        manager._add_new_entities(population, 3, manager.base_titan_affinities, None)
    finally:
        evaluator.close()
    assert len(population) == 3
    if case in ("custom_normalizer", "custom_rng"):
        assert calls == [1, 1, 1]


def test_zero_growth_keeps_distribution_read_but_initial_population_does_not(native):
    manager = make_manager()
    calls = []
    manager.get_global_path_distribution = lambda population: calls.append(1)
    manager._add_new_entities([], 0, manager.base_titan_affinities, None)
    assert calls == [1]
    manager.create_initial_population(0, [], None)
    assert calls == [1]


def test_simulation_newborn_factory_reuses_pool_and_close_lifecycle(native, monkeypatch):
    from simulation import AeonEvolution
    monkeypatch.setitem(config['llm'], 'enable_llm', False)
    monkeypatch.setitem(config['runtime'], 'interactive', False)
    monkeypatch.setitem(config['runtime'], 'torch_device', 'cpu')
    kwargs = {key: value for key, value in config['simulation'].items()
              if key != 'elites_to_keep_in_cycle'}
    kwargs['num_initial_entities'] = 8
    simulation = AeonEvolution(**kwargs, kaoselanna_llm_enabled=False)
    evaluator = None
    try:
        factory = simulation.population_manager._native_evaluator_factory
        assert factory.__self__ is simulation
        assert factory.__func__ is AeonEvolution._get_native_evaluator
        evaluator = factory()
        assert factory() is evaluator
        assert simulation._get_native_evaluator() is evaluator
    finally:
        simulation.close()
    with pytest.raises(RuntimeError, match='closed'):
        evaluator.newborn_affinities(*numerical_inputs())
