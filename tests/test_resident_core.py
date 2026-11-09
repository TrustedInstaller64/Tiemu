"""Contracts for the interpreter-free resident world C ABI."""
import ctypes as ct

import numpy as np
import pytest

from native_backend import configure_backend
from native_encounters import _Config, _Event, _Result, _State, _Trace


class Source(ct.Structure):
    _fields_ = [('stable_id', ct.c_uint64), ('titans', ct.POINTER(ct.c_double)),
                ('paths', ct.POINTER(ct.c_double)), ('score', ct.c_double),
                ('hit_points', ct.c_double), ('trait', ct.c_int32),
                ('aspect', ct.c_int32), ('boss', ct.c_uint8), ('unlocked', ct.c_uint8)]


class View(ct.Structure):
    _fields_ = [('entities', ct.c_size_t), ('stable_ids', ct.POINTER(ct.c_uint64)),
                ('state', _State)]


def pointer(array):
    return None if array is None else array.ctypes.data_as(ct.POINTER(ct.c_double))


@pytest.fixture
def abi():
    native = configure_backend('cpp')
    library = native._lib
    signatures = {
        'tiemu_world_abi_version': ([], ct.c_uint32),
        'tiemu_world_create': ([ct.POINTER(ct.c_void_p)], ct.c_int),
        'tiemu_world_sync': ([ct.c_void_p, ct.c_void_p, ct.c_size_t], ct.c_int),
        'tiemu_world_get_view': ([ct.c_void_p, ct.c_void_p], ct.c_int),
        'tiemu_world_run_encounters': ([ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                       ct.c_void_p, ct.c_size_t, ct.c_void_p], ct.c_int),
        'tiemu_world_births': ([ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                              ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t,
                              ct.c_void_p, ct.c_size_t, ct.c_double, ct.c_double,
                              ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_size_t)], ct.c_int),
        'tiemu_world_destroy': ([ct.POINTER(ct.c_void_p)], ct.c_int),
    }
    try:
        for name, (arguments, result) in signatures.items():
            assert hasattr(library, name), f'Resident C ABI has not implemented {name}'
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, result
        assert library.tiemu_world_abi_version() == 1
        yield native, library
    finally:
        configure_backend('python')


@pytest.fixture
def world(abi):
    _, library = abi
    handle = ct.c_void_p()
    assert library.tiemu_world_create(ct.byref(handle)) == 0
    try:
        yield handle
    finally:
        assert library.tiemu_world_destroy(ct.byref(handle)) == 0
        assert not handle.value
        assert library.tiemu_world_destroy(ct.byref(handle)) == 0


def rows(count=3, ids=None):
    titans = np.arange(count*12, dtype=np.float64).reshape(count, 12) / 10 + 0.25
    paths = np.arange(count*12, dtype=np.float64).reshape(count, 12) / 100 + 0.01
    records = (Source * count)()
    for index in range(count):
        records[index] = Source(0 if ids is None else ids[index], pointer(titans[index]),
                                pointer(paths[index]) if index % 2 else None,
                                1.5 + index, 100.0 - index, index % 3, index - 1,
                                index == 1, index == 2)
    return records, titans, paths


def read(library, handle):
    view = View()
    assert library.tiemu_world_get_view(handle, ct.byref(view)) == 0
    n = int(view.entities)
    return {
        'ids': np.ctypeslib.as_array(view.stable_ids, shape=(n,)).copy() if n else np.empty(0, dtype=np.uint64),
        'titans': np.ctypeslib.as_array(view.state.titans, shape=(n, 12)).copy() if n else np.empty((0, 12)),
        'paths': np.ctypeslib.as_array(view.state.paths, shape=(n, 12)).copy() if n else np.empty((0, 12)),
        'valid': np.ctypeslib.as_array(view.state.path_valid, shape=(n,)).copy() if n else np.empty(0, dtype=np.uint8),
        'scores': np.ctypeslib.as_array(view.state.scores, shape=(n,)).copy() if n else np.empty(0),
    }


def same(actual, expected):
    for field in actual:
        np.testing.assert_array_equal(actual[field], expected[field])


def test_resident_ids_survive_reorder_but_not_removal(abi, world):
    _, library = abi
    source, titans, paths = rows()
    assert library.tiemu_world_sync(world, source, 3) == 0
    initial = read(library, world)
    assert len(set(initial['ids'])) == 3 and all(initial['ids'] > 0)
    reordered, reordered_titans, reordered_paths = rows(2, [int(initial['ids'][2]), int(initial['ids'][0])])
    assert library.tiemu_world_sync(world, reordered, 2) == 0
    np.testing.assert_array_equal(read(library, world)['ids'], initial['ids'][[2, 0]])
    replacement, replacement_titans, replacement_paths = rows(3, [int(initial['ids'][2]), int(initial['ids'][0]), 0])
    assert library.tiemu_world_sync(world, replacement, 3) == 0
    current = read(library, world)
    assert current['ids'][2] > max(initial['ids'])
    stale, stale_titans, stale_paths = rows(1, [int(initial['ids'][1])])
    assert library.tiemu_world_sync(world, stale, 1) == 1
    same(read(library, world), current)


def test_resident_copies_sources_and_resynchronizes_external_writes(abi, world):
    _, library = abi
    source, titans, paths = rows()
    assert library.tiemu_world_sync(world, source, 3) == 0
    initial = read(library, world)
    titans[:] *= 7
    paths[:] *= 3
    same(read(library, world), initial)
    for index, identifier in enumerate(initial['ids']):
        source[index].stable_id = int(identifier)
    assert library.tiemu_world_sync(world, source, 3) == 0
    refreshed = read(library, world)
    np.testing.assert_array_equal(refreshed['titans'], titans)
    np.testing.assert_array_equal(refreshed['paths'][1], paths[1])
    np.testing.assert_array_equal(refreshed['valid'], [0, 1, 0])
    np.testing.assert_array_equal(refreshed['ids'], initial['ids'])


@pytest.mark.parametrize('invalid', ['duplicate', 'unknown', 'null-titan', 'nan-titan',
                                    'nan-path', 'nan-score', 'inf-hp', 'trait', 'aspect', 'boss', 'unlocked'])
def test_invalid_sync_does_not_commit_partial_population(abi, world, invalid):
    _, library = abi
    source, titans, paths = rows()
    assert library.tiemu_world_sync(world, source, 3) == 0
    initial = read(library, world)
    for index, identifier in enumerate(initial['ids']):
        source[index].stable_id = int(identifier)
    source[0].score = 777.0  # a valid earlier row must not commit before a later error
    if invalid == 'duplicate':
        source[2].stable_id = source[0].stable_id
    elif invalid == 'unknown':
        source[2].stable_id = 999999
    elif invalid == 'null-titan':
        source[2].titans = None
    elif invalid == 'nan-titan':
        titans[2, 11] = np.nan
    elif invalid == 'nan-path':
        paths[1, 11] = np.nan
    elif invalid == 'nan-score':
        source[2].score = np.nan
    elif invalid == 'inf-hp':
        source[2].hit_points = np.inf
    elif invalid == 'trait':
        source[2].trait = 9
    elif invalid == 'aspect':
        source[2].aspect = 12
    elif invalid == 'boss':
        source[2].boss = 2
    elif invalid == 'unlocked':
        source[2].unlocked = 2
    assert library.tiemu_world_sync(world, source, 3) == 1
    same(read(library, world), initial)


def test_empty_population_and_handle_contracts(abi, world):
    _, library = abi
    source, titans, paths = rows(1)
    assert library.tiemu_world_sync(world, source, 1) == 0
    old = read(library, world)['ids'][0]
    assert library.tiemu_world_sync(world, None, 0) == 0
    assert read(library, world)['ids'].size == 0
    assert library.tiemu_world_sync(world, source, 1) == 0
    assert read(library, world)['ids'][0] > old
    assert library.tiemu_world_create(ct.byref(world)) == 1
    assert library.tiemu_world_get_view(world, None) == 1
    assert library.tiemu_world_sync(None, source, 1) == 1
    assert library.tiemu_world_sync(world, None, 1) == 1
    assert library.tiemu_world_run_encounters(world, None, None, None, 625, None) == 1
    assert library.tiemu_world_destroy(None) == 1


@pytest.mark.parametrize('modifier_present', [False, True])
@pytest.mark.parametrize('purity', [0.0, 0.01])
def test_native_birth_stage_matches_separate_original_operations(abi, world, modifier_present, purity):
    native, library = abi
    source, titans, paths = rows()
    assert library.tiemu_world_sync(world, source, 3) == 0
    population = read(library, world)
    blueprint = np.linspace(-1.0, 3.0, 12)
    noise = np.linspace(-2.0, 1.0, 60).reshape(5, 12)
    modifier = np.linspace(-0.6, 0.5, 12) if modifier_present else None
    tide = np.linspace(-0.04, 0.03, 12)
    expected = []
    for row in noise:
        values = np.array(blueprint + row).clip(min=0)
        if modifier is not None:
            values += modifier
        expected.append(native.normalize_pipeline(values, purity, tide, 2.5))
    output, count = ct.POINTER(ct.c_double)(), ct.c_size_t()
    status = library.tiemu_world_births(world, blueprint.ctypes.data, blueprint.size,
        noise.ctypes.data, noise.size, len(noise), None if modifier is None else modifier.ctypes.data,
        0 if modifier is None else modifier.size, tide.ctypes.data, tide.size, purity, 2.5,
        ct.byref(output), ct.byref(count))
    assert status == 0 and count.value == noise.size
    actual = np.ctypeslib.as_array(output, shape=noise.shape).copy()
    np.testing.assert_array_equal(actual, expected)
    same(read(library, world), population)


def test_birth_stage_rejects_invalid_shape_before_output(abi, world):
    _, library = abi
    values = np.ones(12)
    output, count = ct.POINTER(ct.c_double)(), ct.c_size_t(999)
    status = library.tiemu_world_births(world, values.ctypes.data, 12, values.ctypes.data,
        11, 1, None, 0, values.ctypes.data, 12, 0.01, 100.0,
        ct.byref(output), ct.byref(count))
    assert status == 1
    assert not output and count.value == 999


@pytest.mark.parametrize('invalid', ['rng-size', 'rng-index', 'population-count',
                                    'reincarnator', 'index', 'result-capacity'])
def test_encounter_failure_does_not_write_inputs_and_complete_sync_recovers(abi, world, invalid):
    native, library = abi
    source, titans, paths = rows()
    assert library.tiemu_world_sync(world, source, 3) == 0
    matrix, relations, modifiers, tide = np.eye(12), np.zeros((12, 12), dtype=np.int32), np.ones((12, 12)), np.zeros(12)
    np_mt, py_mt = np.arange(625, dtype=np.uint32), np.arange(625, dtype=np.uint32)
    culled, updated = np.empty(3, dtype=np.int64), np.empty(3, dtype=np.int64)
    index = native.OpponentIndex([row.score for row in source], 10.0)
    try:
        config = _Config(3, 0, index._live(), pointer(matrix), 144,
                         relations.ctypes.data_as(ct.POINTER(ct.c_int32)), 144,
                         pointer(modifiers), 144, pointer(tide), 12, .01, 100., .1, -1, 0)
        result = _Result()
        result.culled_ids = culled.ctypes.data_as(ct.POINTER(ct.c_int64))
        result.culled_capacity = 3
        result.updated_ids = updated.ctypes.data_as(ct.POINTER(ct.c_int64))
        result.updated_capacity = 3
        words = 625
        if invalid == 'rng-size':
            words = 624
        elif invalid == 'rng-index':
            np_mt[-1] = 625
        elif invalid == 'population-count':
            config.entities = 2
        elif invalid == 'reincarnator':
            config.reincarnator = 3
        elif invalid == 'index':
            config.opponent_index = None
        else:
            result.culled_capacity = 2
        old_np, old_py, old_titans, old_paths = np_mt.copy(), py_mt.copy(), titans.copy(), paths.copy()
        assert library.tiemu_world_run_encounters(world, ct.byref(config), np_mt.ctypes.data,
                                                  py_mt.ctypes.data, words, ct.byref(result)) == 1
        np.testing.assert_array_equal(np_mt, old_np)
        np.testing.assert_array_equal(py_mt, old_py)
        np.testing.assert_array_equal(titans, old_titans)
        np.testing.assert_array_equal(paths, old_paths)
        assert index.alive_count == 3
        # A failing native call is fatal to that attempt, never replayed over
        # its potentially modified index/state. A fresh full sync restores use.
        assert library.tiemu_world_sync(world, source, 3) == 0
        np.testing.assert_array_equal(read(library, world)['titans'], titans)
    finally:
        index.close()


def test_sync_count_overflow_is_rejected_before_source_access(abi, world):
    _, library = abi
    source, titans, paths = rows(1)
    assert library.tiemu_world_sync(world, source, ct.c_size_t(-1).value) == 1
    assert read(library, world)['ids'].size == 0


def test_birth_constructor_clip_preserves_numpy_signed_zero_bits(abi, world):
    native, library = abi
    blueprint, noise, tide = np.full(12, -0.0), np.full((1, 12), -0.0), np.full(12, -0.0)
    blueprint[-1] = 1.0  # Keep the row norm nonzero to expose zero sign bits.
    initial = np.array(blueprint + noise[0]).clip(min=0)
    expected = native.normalize_pipeline(initial, 0., tide, 100.)
    output, count = ct.POINTER(ct.c_double)(), ct.c_size_t()
    assert library.tiemu_world_births(world, blueprint.ctypes.data, 12, noise.ctypes.data,
        12, 1, None, 0, tide.ctypes.data, 12, 0., 100., ct.byref(output), ct.byref(count)) == 0
    actual = np.ctypeslib.as_array(output, shape=(12,)).copy()
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    assert actual.tobytes() == expected.tobytes()


def test_consecutive_resident_segments_refresh_original_array_alias_snapshot(abi, world):
    native, library = abi
    source, titans, paths = rows(2)
    source[0].trait, source[1].trait = 2, 1  # Every encounter boosts the reincarnator.
    # rows() also exercises boss metadata. Disable that here: otherwise the
    # GoldenOne is also its own matching boss and the earlier battle branch wins.
    source[0].boss = source[1].boss = 0
    source[0].aspect = source[1].aspect = -1
    assert library.tiemu_world_sync(world, source, 2) == 0
    matrix, relations, modifiers, tide = np.eye(12), np.zeros((12, 12), dtype=np.int32), np.ones((12, 12)), np.zeros(12)
    np_mt, py_mt = np.zeros(625, dtype=np.uint32), np.zeros(625, dtype=np.uint32)
    np_mt[-1] = py_mt[-1] = 624
    for segment in range(2):
        before = read(library, world)
        index = native.OpponentIndex(before['scores'], 10.0)
        culled, updated, events = np.empty(2, dtype=np.int64), np.empty(2, dtype=np.int64), (_Event * 3)()
        trace = (_Trace * 1)()
        try:
            config = _Config(2, 1, index._live(), pointer(matrix), 144,
                             relations.ctypes.data_as(ct.POINTER(ct.c_int32)), 144,
                             pointer(modifiers), 144, pointer(tide), 12, .01, 100., 0., 0, 0)
            result = _Result()
            result.culled_ids = culled.ctypes.data_as(ct.POINTER(ct.c_int64))
            result.culled_capacity = 2
            result.updated_ids = updated.ctypes.data_as(ct.POINTER(ct.c_int64))
            result.updated_capacity = 2
            result.events, result.event_capacity = events, 3
            result.trace, result.trace_capacity = trace, 1
            assert library.tiemu_world_run_encounters(world, ct.byref(config), np_mt.ctypes.data,
                py_mt.ctypes.data, 625, ct.byref(result)) == 0
            assert result.applied_encounters == 1
            assert result.trace_count == 1 and trace[0].applied == 1 and trace[0].culled == -1
            assert {trace[0].first, trace[0].second} == {0, 1}
            assert result.event_count == 0 and result.updated_count == 2
            view = View()
            assert library.tiemu_world_get_view(world, ct.byref(view)) == 0
            expected_alias = before['titans'].copy()
            expected_alias[0, 0] *= 1.2
            np.testing.assert_array_equal(np.ctypeslib.as_array(view.state.original_titans, shape=(2, 12)), expected_alias)
            assert view.state.original_titan_mutated[0] == 1
            np_mt = np.ctypeslib.as_array(view.state.numpy_mt, shape=(625,)).copy()
            py_mt = np.ctypeslib.as_array(view.state.python_mt, shape=(625,)).copy()
        finally:
            index.close()


def reconcile_function(library):
    assert hasattr(library, 'tiemu_world_reconcile'), 'Resident membership reconciliation has not been implemented'
    function = library.tiemu_world_reconcile
    function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                         ct.c_size_t, ct.c_void_p, ct.c_size_t]
    function.restype = ct.c_int
    return function


def test_membership_reconcile_preserves_ids_and_requires_full_sync_before_numeric_access(abi, world):
    _, library = abi
    reconcile = reconcile_function(library)
    source, titans, paths = rows()
    assert library.tiemu_world_sync(world, source, 3) == 0
    before = read(library, world)
    order = np.array([before['ids'][2], 0, before['ids'][0]], dtype=np.uint64)
    added, added_titans, added_paths = rows(1)
    assigned = np.full(3, 999, dtype=np.uint64)
    assert reconcile(world, order.ctypes.data, 3, added, 1, assigned.ctypes.data, 3) == 0
    assert assigned[0] == before['ids'][2] and assigned[2] == before['ids'][0]
    assert assigned[1] > max(before['ids'])
    view = View()
    assert library.tiemu_world_get_view(world, ct.byref(view)) == 3
    assert library.tiemu_world_run_encounters(world, None, None, None, 625, None) == 3
    # Independent newborn arithmetic remains usable without touching stale rows.
    values, zero = np.ones(12), np.zeros(12)
    output, count = ct.POINTER(ct.c_double)(), ct.c_size_t()
    assert library.tiemu_world_births(world, values.ctypes.data, 12, zero.ctypes.data,
        12, 1, None, 0, zero.ctypes.data, 12, .01, 100., ct.byref(output), ct.byref(count)) == 0
    replacement, replacement_titans, replacement_paths = rows(3, [int(i) for i in assigned])
    replacement_titans[:] *= 4.0  # Import all intervening public writes explicitly.
    assert library.tiemu_world_sync(world, replacement, 3) == 0
    refreshed = read(library, world)
    np.testing.assert_array_equal(refreshed['ids'], assigned)
    np.testing.assert_array_equal(refreshed['titans'], replacement_titans)


@pytest.mark.parametrize('invalid', ['duplicate', 'unknown', 'new-id', 'new-nan',
                                    'missing-new', 'extra-new', 'small-output', 'null-output', 'overflow'])
def test_invalid_reconcile_leaves_live_world_and_result_unchanged(abi, world, invalid):
    _, library = abi
    reconcile = reconcile_function(library)
    source, titans, paths = rows()
    assert library.tiemu_world_sync(world, source, 3) == 0
    before = read(library, world)
    order = np.array([before['ids'][0], 0, before['ids'][2]], dtype=np.uint64)
    added, added_titans, added_paths = rows(1)
    assigned = np.full(3, 999, dtype=np.uint64)
    row_count, new_count, capacity, out_pointer = 3, 1, 3, assigned.ctypes.data
    if invalid == 'duplicate':
        order[2] = order[0]
    elif invalid == 'unknown':
        order[2] = 999999
    elif invalid == 'new-id':
        added[0].stable_id = int(before['ids'][1])
    elif invalid == 'new-nan':
        added_titans[0, 11] = np.nan
    elif invalid == 'missing-new':
        new_count = 0
    elif invalid == 'extra-new':
        order[1] = before['ids'][1]
    elif invalid == 'small-output':
        capacity = 2
    elif invalid == 'null-output':
        out_pointer = None
    else:
        row_count = capacity = ct.c_size_t(-1).value
    assert reconcile(world, order.ctypes.data, row_count, added, new_count, out_pointer, capacity) == 1
    same(read(library, world), before)
    np.testing.assert_array_equal(assigned, [999, 999, 999])


def test_multiple_reconciliations_and_clear_never_reuse_removed_ids(abi, world):
    _, library = abi
    reconcile = reconcile_function(library)
    source, titans, paths = rows(1)
    assert library.tiemu_world_sync(world, source, 1) == 0
    first_id = read(library, world)['ids'][0]
    order, assigned = np.array([0], dtype=np.uint64), np.empty(1, dtype=np.uint64)
    assert reconcile(world, order.ctypes.data, 1, source, 1, assigned.ctypes.data, 1) == 0
    second_id = int(assigned[0])
    assert second_id > first_id
    assert reconcile(world, order.ctypes.data, 1, source, 1, assigned.ctypes.data, 1) == 0
    assert assigned[0] > second_id
    assert reconcile(world, None, 0, None, 0, None, 0) == 0
    assert library.tiemu_world_sync(world, None, 0) == 0
    assert read(library, world)['ids'].size == 0
