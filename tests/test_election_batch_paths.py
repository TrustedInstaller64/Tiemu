"""Remote-only equivalence checks for batched election path selection."""
import copy
import gc
import platform
import sys

import numpy as np
import pytest

from entities import Pathstrider
from models import TitanToPathModel
from native_backend import configure_backend
import parliament_manager as election


@pytest.fixture
def native():
    module = configure_backend('cpp')
    try:
        yield module
    finally:
        configure_backend('python')


def population():
    model = TitanToPathModel()
    result = [Pathstrider(str(i), np.arange(1., 13.) + i, titan_to_path_model=model)
              for i in range(6)]
    for i, entity in enumerate(result):
        entity.score = float(i + 1)
    # Cached paths deliberately differ from the current titan mapping.
    result[0]._path_affinities = np.zeros(12)
    result[2]._path_affinities = np.array([0., 0., 1., 1.] + [0.] * 8)
    result[4]._path_affinities = np.arange(24., dtype=np.float64)[::2]
    result[4]._path_affinities.flags.writeable = False
    return result


def test_batch_maps_only_missing_rows_and_preserves_existing_cache_identity(native, monkeypatch):
    actual = population()
    expected = copy.deepcopy(actual)
    # NaN argmax follows exactly the same NumPy semantics on each row.
    actual[2]._path_affinities[5] = expected[2]._path_affinities[5] = np.nan
    expected_indices = np.array([p.dominant_path_idx for p in expected], dtype=np.int64)
    aliases = [p._path_affinities for p in actual]
    map_paths, calls, outputs = native.path_affinities, [], []

    def record(values, matrix):
        calls.append(values.shape)
        result = map_paths(values, matrix)
        outputs.append(result)
        return result

    monkeypatch.setattr(native, 'path_affinities', record)
    np.testing.assert_array_equal(election._native_dominant_paths(actual, native), expected_indices)
    assert calls == [(3, 12)]
    for index, (entity, reference) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(entity._path_affinities, reference._path_affinities)
        if aliases[index] is not None:
            assert entity._path_affinities is aliases[index]
        else:
            assert entity._path_affinities.flags.owndata
            assert not np.shares_memory(entity._path_affinities, outputs[0])
    # A second election reuses every cache and makes no mapping calls.
    filled_aliases = [p._path_affinities for p in actual]
    np.testing.assert_array_equal(election._native_dominant_paths(actual, native), expected_indices)
    assert calls == [(3, 12)]
    assert all(p._path_affinities is old for p, old in zip(actual, filled_aliases))


def test_election_matches_original_scalar_native_path_and_cache_values(native, monkeypatch):
    actual = population()
    expected = copy.deepcopy(actual)
    scalar, batch = election.ParliamentManager(), election.ParliamentManager()
    with monkeypatch.context() as patch:
        patch.setattr(election, '_native_dominant_paths', lambda *_: None)
        scalar.hold_election(expected)
    batch.hold_election(actual)
    assert batch.seats == scalar.seats
    for entity, reference in zip(actual, expected):
        np.testing.assert_array_equal(entity._path_affinities, reference._path_affinities)


@pytest.mark.parametrize('case', [
    'entity_subclass', 'model_subclass', 'instance_override', 'different_mapping',
    'duplicate', 'titan_dtype', 'titan_shape', 'cache_dtype', 'cache_shape',
    'matrix_shape', 'path_property', 'dominant_property', 'model_method',
])
def test_unsupported_population_falls_back_before_any_cache_fill(native, monkeypatch, case):
    actual = population()
    model = actual[0]._titan_to_path_model
    if case == 'entity_subclass':
        class CustomEntity(Pathstrider):
            pass
        actual[-1] = CustomEntity('custom', np.ones(12), titan_to_path_model=model)
    elif case == 'model_subclass':
        class CustomModel(TitanToPathModel):
            pass
        actual[-1]._titan_to_path_model = CustomModel()
    elif case == 'instance_override':
        model.get_path_affinities = lambda _: np.zeros(12)
    elif case == 'different_mapping':
        actual[-1]._titan_to_path_model = TitanToPathModel()
    elif case == 'duplicate':
        actual.append(actual[0])
    elif case == 'titan_dtype':
        actual[-1].titan_affinities = np.ones(12, dtype=np.float32)
    elif case == 'titan_shape':
        actual[-1].titan_affinities = np.ones((1, 12))
    elif case == 'cache_dtype':
        actual[-1]._path_affinities = np.ones(12, dtype=np.float32)
    elif case == 'cache_shape':
        actual[-1]._path_affinities = np.ones((1, 12))
    elif case == 'matrix_shape':
        model.titan_to_path_matrix = np.eye(3)
    elif case == 'path_property':
        monkeypatch.setattr(Pathstrider, 'path_affinities', property(lambda _: np.zeros(12)))
    elif case == 'dominant_property':
        monkeypatch.setattr(Pathstrider, 'dominant_path_idx', property(lambda _: 0))
    elif case == 'model_method':
        monkeypatch.setattr(TitanToPathModel, 'get_path_affinities', lambda *_: np.zeros(12))
    aliases = [p._path_affinities for p in actual]
    monkeypatch.setattr(native, 'path_affinities', lambda *_: pytest.fail('unexpected mapping'))
    assert election._native_dominant_paths(actual, native) is None
    assert all(p._path_affinities is old for p, old in zip(actual, aliases))


@pytest.mark.skipif(platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 12),
                    reason='This mechanism contract targets CPython 3.12 managed instance dictionaries')
def test_batch_path_access_does_not_materialize_entity_dictionaries(native):
    # A2's entity.__dict__ reads changed later attribute-scan performance even
    # though all numerical state stayed equal. Test the object-layout side
    # effect directly, without a timing threshold or private CPython memory API.
    def has_dict(entity):
        return any(type(referent) is dict for referent in gc.get_referents(entity))

    actual = population()
    assert not any(has_dict(entity) for entity in actual)
    control = Pathstrider('control', np.ones(12), titan_to_path_model=actual[0]._titan_to_path_model)
    assert not has_dict(control)
    control.__dict__  # Confirm the public GC observation detects materialization.
    assert has_dict(control)
    indices = election._native_dominant_paths(actual, native)
    assert indices is not None and indices.shape == (len(actual),)
    assert not any(has_dict(entity) for entity in actual)
    # Already-populated caches must stay on the same safe access path too.
    np.testing.assert_array_equal(election._native_dominant_paths(actual, native), indices)
    assert not any(has_dict(entity) for entity in actual)


@pytest.mark.parametrize('field', ['_titan_to_path_model', '_path_affinities', 'titan_affinities'])
def test_batch_path_field_descriptors_fall_back_without_invoking_them(native, monkeypatch, field):
    actual = population()
    aliases = [entity._path_affinities for entity in actual]
    accesses = []

    class FieldDescriptor:
        def __get__(self, instance, owner):
            accesses.append(('get', instance))
            raise AssertionError('native eligibility must not invoke a field descriptor')

        def __set__(self, instance, value):
            accesses.append(('set', instance))
            raise AssertionError('native eligibility must not write through a field descriptor')

    with monkeypatch.context() as patch:
        patch.setattr(Pathstrider, field, FieldDescriptor(), raising=False)
        assert election._native_dominant_paths(actual, native) is None
    assert accesses == []
    assert all(entity._path_affinities is old for entity, old in zip(actual, aliases))


def test_missing_field_with_custom_getattr_falls_back_without_callback(native, monkeypatch):
    actual = population()
    aliases = [entity._path_affinities for entity in actual]
    del actual[-1]._titan_to_path_model
    visits = []

    def custom_getattr(entity, name):
        visits.append(name)
        raise AssertionError('batch eligibility must not invoke custom fallback attribute access')

    monkeypatch.setattr(Pathstrider, '__getattr__', custom_getattr, raising=False)
    assert election._native_dominant_paths(actual, native) is None
    assert visits == []
    assert all(entity._path_affinities is old for entity, old in zip(actual, aliases))
