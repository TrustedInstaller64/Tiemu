"""Contracts for the standalone resident scoring preparation ABI.

These tests call the C exports directly and treat the existing numerical core
as the oracle; no tolerance is added for migration.
"""
import ctypes as ct

import numpy as np
import pytest

from native_backend import configure_backend
from native_encounters import _Config, _Result, _State


P = ct.POINTER(ct.c_double)


class ScoringSource(ct.Structure):
    _fields_ = [('titans', P), ('cached_paths', P), ('boss', ct.c_uint8)]


class ScoringView(ct.Structure):
    _fields_ = [('rows', ct.c_size_t), ('token', ct.c_uint64),
                ('values', P), ('old_paths', P),
                ('bosses', ct.POINTER(ct.c_int64)),
                ('repaired', ct.POINTER(ct.c_uint8)), ('paths', P), ('stats', P)]


class WorldSource(ct.Structure):
    _fields_ = [('stable_id', ct.c_uint64), ('titans', P), ('paths', P),
                ('score', ct.c_double), ('hit_points', ct.c_double),
                ('trait', ct.c_int32), ('aspect', ct.c_int32),
                ('boss', ct.c_uint8), ('unlocked', ct.c_uint8)]


class WorldView(ct.Structure):
    _fields_ = [('entities', ct.c_size_t), ('stable_ids', ct.POINTER(ct.c_uint64)),
                ('state', _State)]


def ptr(array):
    return None if array is None else array.ctypes.data_as(P)


def copy_array(pointer, shape, dtype=np.float64):
    if not np.prod(shape):
        return np.empty(shape, dtype=dtype)
    return np.ctypeslib.as_array(pointer, shape=shape).copy()


def view_bytes(view):
    return ct.string_at(ct.byref(view), ct.sizeof(view))


@pytest.fixture
def scoring_abi():
    native = configure_backend('cpp')
    library = native._lib
    signatures = {
        'tiemu_world_create': [ct.c_void_p],
        'tiemu_world_destroy': [ct.c_void_p],
        'tiemu_world_sync': [ct.c_void_p, ct.c_void_p, ct.c_size_t],
        'tiemu_world_get_view': [ct.c_void_p, ct.c_void_p],
        'tiemu_world_reconcile': [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                                  ct.c_size_t, ct.c_void_p, ct.c_size_t],
        'tiemu_world_run_encounters': [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                       ct.c_void_p, ct.c_size_t, ct.c_void_p],
        'tiemu_world_births': [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                              ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t,
                              ct.c_void_p, ct.c_size_t, ct.c_double, ct.c_double,
                              ct.c_void_p, ct.c_void_p],
        'tiemu_world_scoring_prepare': [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                        ct.c_void_p, ct.c_size_t, ct.c_void_p],
        'tiemu_world_scoring_finish': [ct.c_void_p, ct.c_uint64, ct.c_void_p,
                                       ct.c_void_p, ct.c_size_t, ct.c_void_p,
                                       ct.c_size_t, ct.c_void_p],
    }
    try:
        for name, arguments in signatures.items():
            assert hasattr(library, name), f'Resident scoring C ABI has not implemented {name}'
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, ct.c_int
        yield native, library
    finally:
        configure_backend('python')


@pytest.fixture
def scoring_world(scoring_abi):
    _, library = scoring_abi
    handle = ct.c_void_p()
    assert library.tiemu_world_create(ct.byref(handle)) == 0
    values = np.arange(1., 25.).reshape(2, 12)
    sources = (WorldSource * 2)()
    for index in range(2):
        sources[index] = WorldSource(0, ptr(values[index]), None, 3. + index,
                                     100., 0, -1, 0, 0)
    assert library.tiemu_world_sync(handle, sources, 2) == 0
    initial = world_snapshot(library, handle)
    for index, identifier in enumerate(initial['ids']):
        sources[index].stable_id = int(identifier)
    try:
        yield handle, sources, values, initial
    finally:
        assert library.tiemu_world_destroy(ct.byref(handle)) == 0
        assert not handle.value
        assert library.tiemu_world_destroy(ct.byref(handle)) == 0


def world_snapshot(library, handle):
    view = WorldView()
    assert library.tiemu_world_get_view(handle, ct.byref(view)) == 0
    n = view.entities
    return {'ids': copy_array(view.stable_ids, (n,), np.uint64),
            'values': copy_array(view.state.titans, (n, 12)),
            'scores': copy_array(view.state.scores, (n,))}


def assert_world_same(actual, expected):
    for field in expected:
        np.testing.assert_array_equal(actual[field], expected[field])


def scoring_sources(nonfinite=True):
    values = np.arange(1., 49.).reshape(4, 12) / 11.
    cached = np.arange(1., 49.).reshape(4, 12)[:, ::-1].copy() / 53.
    values[-1] = 0.
    if nonfinite:
        values[1, 3] = np.nan  # Missing old cache: map before repairing this row.
        values[2, 7] = np.inf  # Stale finite cache must win for the old-path input.
    sources = (ScoringSource * 4)()
    for index in range(4):
        sources[index] = ScoringSource(ptr(values[index]), ptr(cached[index]) if index % 2 == 0 else None,
                                       index == 2)
    matrix = np.arange(1., 145.).reshape(12, 12) / 149.
    return sources, values, cached, matrix


def prepare(library, handle, sources, matrix):
    view = ScoringView()
    assert library.tiemu_world_scoring_prepare(handle, sources, len(sources),
                                              matrix.ctypes.data, matrix.size, ct.byref(view)) == 0
    assert view.token and view.rows == len(sources)
    assert not view.paths and not view.stats
    return view


def finish(library, handle, token, evaluator, multipliers, distribution=None):
    result = ScoringView()
    assert library.tiemu_world_scoring_finish(handle, token, evaluator._live(),
        multipliers.ctypes.data, multipliers.size,
        None if distribution is None else distribution.ctypes.data,
        0 if distribution is None else distribution.size, ct.byref(result)) == 0
    return result


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('distribution', [None, np.arange(1., 13.) / 78.])
def test_scoring_copies_old_cache_before_repair_and_reuses_original_batch_kernel(scoring_abi, scoring_world,
                                                                               workers, distribution):
    native, library = scoring_abi
    handle, live_sources, live_values, initial = scoring_world
    sources, values, cached, matrix = scoring_sources()
    expected_values = values.copy()
    expected_old = np.array([cached[i].copy() if i % 2 == 0 else native.path_affinities(row, matrix)
                             for i, row in enumerate(values)])
    repaired = ~np.isfinite(values).all(axis=1)
    expected_values[repaired] = 1.
    multipliers = np.array([.91, .98, 1.07, 1.19])
    bosses = np.array([0, 0, 1, 0], dtype=np.int64)
    original_values, original_cache, original_matrix = values.copy(), cached.copy(), matrix.copy()
    evaluator = native.BatchEvaluator(workers=workers, min_rows=1)
    try:
        expected_paths, expected_stats = evaluator.concepts(expected_values, matrix, multipliers,
                                                            distribution, bosses)
        view = prepare(library, handle, sources, matrix)
        np.testing.assert_array_equal(copy_array(view.values, (4, 12)), expected_values)
        np.testing.assert_array_equal(copy_array(view.old_paths, (4, 12)), expected_old)
        np.testing.assert_array_equal(copy_array(view.repaired, (4,), np.uint8), repaired)
        np.testing.assert_array_equal(copy_array(view.bosses, (4,), np.int64), bosses)
        np.testing.assert_array_equal(values, original_values)
        np.testing.assert_array_equal(cached, original_cache)
        np.testing.assert_array_equal(matrix, original_matrix)
        assert library.tiemu_world_get_view(handle, ct.byref(WorldView())) == 3
        assert library.tiemu_world_run_encounters(handle, None, None, None, 625, None) == 3
        # No caller-owned input pointer is retained by preparation.
        values[:] = 777.
        cached[:] = -999.
        matrix[:] = 0.
        result = finish(library, handle, view.token, evaluator, multipliers, distribution)
        assert result.token == view.token and result.rows == 4
        np.testing.assert_array_equal(copy_array(result.values, (4, 12)), expected_values)
        np.testing.assert_array_equal(copy_array(result.old_paths, (4, 12)), expected_old)
        np.testing.assert_array_equal(copy_array(result.paths, (4, 12)), expected_paths)
        np.testing.assert_array_equal(copy_array(result.stats, (4, 4)), expected_stats)
        assert library.tiemu_world_get_view(handle, ct.byref(WorldView())) == 3
        assert library.tiemu_world_sync(handle, live_sources, 2) == 0
        assert_world_same(world_snapshot(library, handle), initial)
    finally:
        evaluator.close()


@pytest.mark.parametrize('invalid', ['null-handle', 'null-sources', 'overflow', 'matrix-size',
                                    'null-matrix', 'null-titan', 'boss', 'null-output'])
def test_invalid_prepare_preserves_live_state_old_preparation_and_output(scoring_abi, scoring_world, invalid):
    native, library = scoring_abi
    handle, live_sources, live_values, initial = scoring_world
    sources, values, cached, matrix = scoring_sources(nonfinite=False)
    prepared = None
    evaluator = native.BatchEvaluator()
    try:
        # Check the same rejection both before and after an existing preparation.
        for already_prepared in (False, True):
            if already_prepared:
                prepared = prepare(library, handle, sources, matrix)
            arguments = [handle, sources, 4, matrix.ctypes.data, 144]
            result = ScoringView(rows=999, token=123)
            original_result = view_bytes(result)
            output = ct.byref(result)
            if invalid == 'null-handle':
                arguments[0] = None
            elif invalid == 'null-sources':
                arguments[1] = None
            elif invalid == 'overflow':
                arguments[2] = ct.c_size_t(-1).value
            elif invalid == 'matrix-size':
                arguments[4] = 143
            elif invalid == 'null-matrix':
                arguments[3] = None
            elif invalid == 'null-titan':
                sources[-1].titans = None
            elif invalid == 'boss':
                sources[-1].boss = 2
            else:
                output = None
            assert library.tiemu_world_scoring_prepare(*arguments, output) == 1
            assert view_bytes(result) == original_result
            sources[-1].titans, sources[-1].boss = ptr(values[-1]), 0
            if not already_prepared:
                assert_world_same(world_snapshot(library, handle), initial)
            else:
                assert library.tiemu_world_get_view(handle, ct.byref(WorldView())) == 3
                result = finish(library, handle, prepared.token, evaluator, np.ones(4))
                np.testing.assert_array_equal(copy_array(result.values, (4, 12)), values)
    finally:
        evaluator.close()


@pytest.mark.parametrize('invalid', ['token', 'null-evaluator', 'closed-evaluator', 'multiplier-count',
                                    'null-multipliers', 'distribution-count', 'null-distribution', 'null-output'])
def test_invalid_finish_never_publishes_output_and_new_prepare_recovers(scoring_abi, scoring_world, invalid):
    native, library = scoring_abi
    handle, _, _, _ = scoring_world
    sources, values, cached, matrix = scoring_sources(nonfinite=False)
    prepared = prepare(library, handle, sources, matrix)
    evaluator = native.BatchEvaluator()
    multipliers, distribution = np.ones(4), np.ones(12) / 12.
    result = ScoringView(rows=999, token=123)
    before = view_bytes(result)
    arguments = [handle, prepared.token, evaluator._live(), multipliers.ctypes.data, 4,
                 distribution.ctypes.data, 12, ct.byref(result)]
    expected_status = 1
    if invalid == 'token':
        arguments[1] = prepared.token + 1
        expected_status = 3
    elif invalid == 'null-evaluator':
        arguments[2] = None
    elif invalid == 'closed-evaluator':
        evaluator.close()
        expected_status = 3
    elif invalid == 'multiplier-count':
        arguments[4] = 3
    elif invalid == 'null-multipliers':
        arguments[3] = None
    elif invalid == 'distribution-count':
        arguments[6] = 11
    elif invalid == 'null-distribution':
        arguments[5] = None
    else:
        arguments[7] = None
    try:
        assert library.tiemu_world_scoring_finish(*arguments) == expected_status
        assert view_bytes(result) == before
    finally:
        evaluator.close()
    replacement = native.BatchEvaluator()
    try:
        renewed = prepare(library, handle, sources, matrix)
        assert renewed.token != prepared.token
        assert finish(library, handle, renewed.token, replacement, multipliers).rows == 4
    finally:
        replacement.close()


def run_empty_encounters(native, library, handle, count, *, fail=False):
    matrix, relations, modifiers, tide = np.eye(12), np.zeros((12, 12), dtype=np.int32), np.ones((12, 12)), np.zeros(12)
    states = np.zeros(625, dtype=np.uint32)
    states[-1] = 624
    output_ids = np.empty((2, count), dtype=np.int64)
    index = native.OpponentIndex(np.arange(count, dtype=np.float64) + 3., 10.)
    try:
        config = _Config(count, 0, None if fail else index._live(), ptr(matrix), 144,
                         relations.ctypes.data_as(ct.POINTER(ct.c_int32)), 144,
                         ptr(modifiers), 144, ptr(tide), 12, .01, 100., .1, -1, 0)
        result = _Result()
        result.culled_ids = output_ids[0].ctypes.data_as(ct.POINTER(ct.c_int64))
        result.updated_ids = output_ids[1].ctypes.data_as(ct.POINTER(ct.c_int64))
        result.culled_capacity = result.updated_capacity = count
        return library.tiemu_world_run_encounters(handle, ct.byref(config), states.ctypes.data,
                                                 states.ctypes.data, 625, ct.byref(result))
    finally:
        index.close()


@pytest.mark.parametrize('mutation', ['prepare', 'sync', 'reconcile', 'births', 'run'])
def test_successful_world_mutation_invalidates_old_scoring_token(scoring_abi, scoring_world, mutation):
    native, library = scoring_abi
    handle, live_sources, live_values, initial = scoring_world
    sources, values, cached, matrix = scoring_sources(nonfinite=False)
    prepared = prepare(library, handle, sources, matrix)
    evaluator = native.BatchEvaluator()
    multipliers = np.ones(4)
    try:
        if mutation == 'prepare':
            assert prepare(library, handle, sources, matrix).token != prepared.token
        elif mutation in ('sync', 'run'):
            assert library.tiemu_world_sync(handle, live_sources, 2) == 0
            if mutation == 'run':
                assert run_empty_encounters(native, library, handle, 2) == 0
        elif mutation == 'reconcile':
            order, assigned = initial['ids'].copy(), np.empty(2, dtype=np.uint64)
            assert library.tiemu_world_reconcile(handle, order.ctypes.data, 2, None, 0,
                                                 assigned.ctypes.data, 2) == 0
            np.testing.assert_array_equal(assigned, initial['ids'])
        else:
            blueprint, noise, tide = np.ones(12), np.zeros((1, 12)), np.zeros(12)
            output, count = P(), ct.c_size_t()
            assert library.tiemu_world_births(handle, blueprint.ctypes.data, 12, noise.ctypes.data,
                12, 1, None, 0, tide.ctypes.data, 12, .01, 100., ct.byref(output), ct.byref(count)) == 0
        output = ScoringView(rows=999, token=123)
        before = view_bytes(output)
        assert library.tiemu_world_scoring_finish(handle, prepared.token, evaluator._live(),
            multipliers.ctypes.data, 4, None, 0, ct.byref(output)) == 3
        assert view_bytes(output) == before
    finally:
        evaluator.close()


def test_failed_encounter_requires_sync_before_scoring_but_membership_dirty_does_not(scoring_abi, scoring_world):
    native, library = scoring_abi
    handle, live_sources, live_values, initial = scoring_world
    sources, values, cached, matrix = scoring_sources(nonfinite=False)
    assert run_empty_encounters(native, library, handle, 2, fail=True) == 1
    output = ScoringView(rows=999, token=123)
    before = view_bytes(output)
    assert library.tiemu_world_scoring_prepare(handle, sources, 4, matrix.ctypes.data, 144,
                                              ct.byref(output)) == 3
    assert view_bytes(output) == before
    assert library.tiemu_world_sync(handle, live_sources, 2) == 0
    order, assigned = initial['ids'].copy(), np.empty(2, dtype=np.uint64)
    assert library.tiemu_world_reconcile(handle, order.ctypes.data, 2, None, 0,
                                         assigned.ctypes.data, 2) == 0
    prepared = prepare(library, handle, sources, matrix)
    evaluator = native.BatchEvaluator()
    try:
        assert finish(library, handle, prepared.token, evaluator, np.ones(4)).rows == 4
    finally:
        evaluator.close()


def test_empty_scoring_single_use_and_owned_copies_survive_reuse_and_close(scoring_abi, scoring_world):
    native, library = scoring_abi
    handle, live_sources, live_values, initial = scoring_world
    matrix, output = np.eye(12), ScoringView()
    evaluator = native.BatchEvaluator()
    try:
        assert library.tiemu_world_scoring_prepare(handle, None, 0, matrix.ctypes.data, 144,
                                                  ct.byref(output)) == 0
        assert output.rows == 0 and output.token
        result = ScoringView()
        assert library.tiemu_world_scoring_finish(handle, output.token, evaluator._live(),
                                                 None, 0, None, 0, ct.byref(result)) == 0
        assert result.rows == 0
        assert library.tiemu_world_scoring_finish(handle, output.token, evaluator._live(),
                                                 None, 0, None, 0, ct.byref(result)) == 3
        sources, values, cached, matrix = scoring_sources(nonfinite=False)
        prepared = prepare(library, handle, sources, matrix)
        result = finish(library, handle, prepared.token, evaluator, np.ones(4))
        owned = copy_array(result.paths, (4, 12)), copy_array(result.stats, (4, 4))
        expected = tuple(array.copy() for array in owned)
        values[:] *= 2.
        replacement = prepare(library, handle, sources, matrix)
        finish(library, handle, replacement.token, evaluator, np.full(4, 1.2))
        assert library.tiemu_world_sync(handle, live_sources, 2) == 0
        assert_world_same(world_snapshot(library, handle), initial)
        assert library.tiemu_world_destroy(ct.byref(handle)) == 0
        sentinel = ScoringView(rows=999, token=123)
        before = view_bytes(sentinel)
        assert library.tiemu_world_scoring_prepare(handle, sources, 4, matrix.ctypes.data, 144,
                                                  ct.byref(sentinel)) == 1
        assert view_bytes(sentinel) == before
        for actual, reference in zip(owned, expected):
            np.testing.assert_array_equal(actual, reference)
    finally:
        evaluator.close()
