"""A3 resident-world contracts. Execute only on the hosted cloud runners."""
import gc
import importlib
import random
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from entities import Pathstrider
from models import TitanToPathModel
from native_backend import configure_backend


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
        pytest.fail(f'Resident C++ world has not been implemented: {exc}')


def population(size=4):
    model = TitanToPathModel()
    values = []
    for index in range(size):
        entity = Pathstrider(f'entity-{index}', np.arange(1., 13.) + index,
                            titan_to_path_model=model)
        entity.score = float(index + 1)
        if index % 2:
            entity._path_affinities = np.roll(np.arange(1., 13.) / 78., index).copy()
        values.append(entity)
    return values


def assert_snapshot_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def test_resident_ids_survive_reorder_and_new_ids_do_not_reuse_deleted_slots(native):
    world = resident_type()()
    entities = population()
    try:
        world.sync(entities)
        first_ids = world.stable_ids()
        assert len(set(first_ids)) == 4 and min(first_ids) > 0
        world.sync([entities[2], entities[0], entities[3]])
        assert world.stable_ids() == (first_ids[2], first_ids[0], first_ids[3])
        newcomer = population(1)[0]
        newcomer.name = entities[1].name  # A name is not an entity identity.
        world.sync([entities[2], newcomer, entities[0], entities[3]])
        inserted = world.stable_ids()
        assert inserted[0] == first_ids[2] and inserted[2:] == (first_ids[0], first_ids[3])
        assert inserted[1] > max(first_ids)
        world.sync([entities[1]])  # The old removed object is a new admission.
        assert world.stable_ids()[0] > max(inserted)
        world.sync([])
        assert world.stable_ids() == ()
    finally:
        world.close()


def test_sync_observes_inplace_array_scalar_and_stale_cache_changes(native):
    world = resident_type()()
    entities = population()
    try:
        world.sync(entities)
        first_ids = world.stable_ids()
        entities[0].titan_affinities[:] *= 3.0
        entities[0].score = 99.5
        entities[0].hp = np.float64(17.)
        entities[1]._path_affinities[:] = np.arange(12.) / 100.
        entities[2].trait = 'GoldenOne'
        entities[2].data_modification_unlocked = True
        world.sync(entities)
        saved = world.snapshot()
        assert world.stable_ids() == first_ids
        np.testing.assert_array_equal(saved['titans'], np.stack([p.titan_affinities for p in entities]))
        np.testing.assert_array_equal(saved['scores'], [p.score for p in entities])
        np.testing.assert_array_equal(saved['hp'], [p.hp for p in entities])
        np.testing.assert_array_equal(saved['paths'][1], entities[1]._path_affinities)
        assert saved['path_valid'].tolist() == [0, 1, 0, 1]
        assert entities[0]._path_affinities is None
        assert saved['unlocked'].tolist() == [0, 0, 1, 0]
    finally:
        world.close()


@pytest.mark.parametrize('case', ['duplicate', 'wrong_shape', 'nonfinite', 'shared', 'readonly'])
def test_invalid_sync_keeps_last_committed_resident_state(native, case):
    world = resident_type()()
    entities = population()
    try:
        world.sync(entities)
        before = world.snapshot()
        ids = world.stable_ids()
        invalid = population()
        if case == 'duplicate':
            invalid[1] = invalid[0]
        elif case == 'wrong_shape':
            invalid[1].titan_affinities = np.ones(11)
        elif case == 'nonfinite':
            invalid[1].titan_affinities[3] = np.nan
        elif case == 'shared':
            invalid[1].titan_affinities = invalid[0].titan_affinities
        else:
            invalid[1].titan_affinities.flags.writeable = False
        with pytest.raises(ValueError):
            world.sync(invalid)
        assert world.stable_ids() == ids
        assert_snapshot_equal(world.snapshot(), before)
    finally:
        world.close()


def test_snapshots_own_their_data_and_world_keeps_entities_alive_only_until_next_sync(native):
    world = resident_type()()
    entities = population()
    reference = weakref.ref(entities[0])
    world.sync(entities)
    before = world.snapshot()
    saved = {name: value.copy() for name, value in before.items()}
    del entities
    gc.collect()
    assert reference() is not None
    world.sync([])
    gc.collect()
    assert reference() is None
    world.close()
    world.close()
    assert_snapshot_equal(before, saved)
    with pytest.raises(RuntimeError):
        world.sync(population())


def test_repeated_native_encounters_use_the_same_resident_world_and_match_frozen_loop(native, monkeypatch):
    from test_native_encounters import (make_world, frozen_oracle, trace_recorder,
                                        assert_same_entities, ids, close_index, rng_state)
    from native_encounters import try_run_encounters
    cls = resident_type()
    calls = []
    original = cls.run_encounters

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, 'run_encounters', counted)
    actual, expected = make_world(relation='REPULSION'), make_world(relation='REPULSION', frozen=True)
    resident = None
    try:
        for generation in range(3):
            # Direct public writes between calls must be imported even though
            # both the entity and its ndarray retain their original identity.
            for current in (actual, expected):
                current.population[0].titan_affinities[2] += generation * 0.02
                current.population[2].score += generation * 0.01
            random.seed(731 + generation)
            np.random.seed(731 + generation)
            culled, updated, trace = frozen_oracle()[1](expected, trace_recorder(expected.population))
            expected_rng = rng_state()
            random.seed(731 + generation)
            np.random.seed(731 + generation)
            result = try_run_encounters(actual, trace=True)
            assert result is not None
            assert result.trace == trace and rng_state() == expected_rng
            assert ids(actual.population, result.culled) == ids(expected.population, culled)
            assert ids(actual.population, result.updated) == ids(expected.population, updated)
            assert_same_entities(actual, expected)
            current = actual.population_manager._native_world
            assert current is not None
            if resident is None:
                resident = current
            assert current is resident
        assert calls == [resident, resident, resident]
    finally:
        close_index(actual)
        close_index(expected)
        if resident is not None:
            resident.close()


def test_simulation_close_releases_resident_state_after_partial_setup(native):
    from simulation import AeonEvolution
    from population_manager import PopulationManager
    model = TitanToPathModel()
    manager = PopulationManager(set(), {}, np.ones(12), .1, .01, 12, .1,
                                model, 100., 1., 1000.)
    assert hasattr(manager, 'get_native_world'), 'Population lifecycle does not own a native world yet'
    world = manager.get_native_world()
    world.sync(population())
    simulation = object.__new__(AeonEvolution)
    simulation.population_manager = manager
    simulation._native_evaluator = None
    simulation.debugger = SimpleNamespace(close=lambda: None)
    simulation.close()
    simulation.close()
    assert manager._native_world is None
    with pytest.raises(RuntimeError):
        world.sync([])
