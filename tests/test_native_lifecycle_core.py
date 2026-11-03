"""Cloud-only, independent ctypes contracts for resident golden metadata plans."""
import ctypes as ct

import numpy as np
import pytest

from native_backend import configure_backend
from native_encounters import _Config, _Result
from native_world import _Source, _View, _ScoringView


class GoldenSource(ct.Structure):
    _fields_ = [('score', ct.c_double), ('trait', ct.c_int32), ('boss', ct.c_uint8)]


class GoldenView(ct.Structure):
    _fields_ = [('reset_count', ct.c_size_t), ('reset_positions', ct.POINTER(ct.c_size_t)),
                ('selected_count', ct.c_size_t), ('selected_positions', ct.POINTER(ct.c_size_t))]


@pytest.fixture
def golden_abi():
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
        'tiemu_world_scoring_prepare': [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                        ct.c_void_p, ct.c_size_t, ct.c_void_p],
        'tiemu_world_scoring_finish': [ct.c_void_p, ct.c_uint64, ct.c_void_p,
                                       ct.c_void_p, ct.c_size_t, ct.c_void_p,
                                       ct.c_size_t, ct.c_void_p],
        'tiemu_world_golden_plan': [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                    ct.c_int64, ct.c_void_p],
    }
    try:
        for name, arguments in signatures.items():
            assert hasattr(library, name), f'Resident golden C ABI has not implemented {name}'
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, ct.c_int
        yield native, library
    finally:
        configure_backend('python')


@pytest.fixture
def golden_world(golden_abi):
    _, library = golden_abi
    handle = ct.c_void_p()
    assert library.tiemu_world_create(ct.byref(handle)) == 0
    values = np.arange(1., 25.).reshape(2, 12)
    sources = (_Source * 2)()
    for index in range(2):
        sources[index] = _Source(0, values[index].ctypes.data, None, 3. + index,
                                  100., 0, -1, 0, 0)
    assert library.tiemu_world_sync(handle, sources, 2) == 0
    view = _View()
    assert library.tiemu_world_get_view(handle, ct.byref(view)) == 0
    ids = np.ctypeslib.as_array(view.stable_ids, shape=(2,)).copy()
    for index, identifier in enumerate(ids):
        sources[index].stable_id = int(identifier)
    try:
        yield handle, sources, values, ids
    finally:
        assert library.tiemu_world_destroy(ct.byref(handle)) == 0
        assert not handle.value
        assert library.tiemu_world_destroy(ct.byref(handle)) == 0


def sources_for(count):
    scores = (5., -0., 0., 5., 12., 12., -7., 1.)
    result = (GoldenSource * count)()
    for index in range(count):
        trait = 1 if index % 4 == 0 else (2 if index % 7 == 1 else 0)
        result[index] = GoldenSource(scores[index % len(scores)], trait, index % 6 == 0)
    return result


def expected_plan(sources, cap):
    reset = [index for index, row in enumerate(sources) if row.trait == 1]
    candidates = [index for index, row in enumerate(sources) if not row.boss and row.trait != 2]
    candidates.sort(key=lambda index: sources[index].score, reverse=True)
    return reset, candidates[:max(12, cap)]


def copied_plan(view):
    return (list(view.reset_positions[:view.reset_count]) if view.reset_count else [],
            list(view.selected_positions[:view.selected_count]) if view.selected_count else [])


def bytes_of(value):
    return ct.string_at(ct.byref(value), ct.sizeof(value))


def plan(library, handle, sources, cap=20):
    output = GoldenView()
    assert library.tiemu_world_golden_plan(handle, sources, len(sources), cap, ct.byref(output)) == 0
    return output


@pytest.mark.parametrize('count', [0, 1, 5, 31])
@pytest.mark.parametrize('cap', [-100, 3, 12, 20, 100])
def test_golden_plan_stably_filters_current_metadata_and_preserves_sources(golden_abi, golden_world, count, cap):
    _, library = golden_abi
    handle, live_sources, live_values, ids = golden_world
    sources = sources_for(count)
    before = bytes_of(sources)
    expected = expected_plan(sources, cap)
    output = plan(library, handle, sources, cap)
    assert copied_plan(output) == expected
    assert bytes_of(sources) == before
    # Even though plan rows differ from live membership, IDs are never replaced.
    assert library.tiemu_world_get_view(handle, ct.byref(_View())) == 3
    assert library.tiemu_world_sync(handle, live_sources, 2) == 0
    restored = _View()
    assert library.tiemu_world_get_view(handle, ct.byref(restored)) == 0
    np.testing.assert_array_equal(np.ctypeslib.as_array(restored.stable_ids, shape=(2,)), ids)
    np.testing.assert_array_equal(np.ctypeslib.as_array(restored.state.titans, shape=(24,)), live_values.ravel())


def test_golden_plan_resets_old_bosses_and_keeps_empty_candidate_plan(golden_abi, golden_world):
    _, library = golden_abi
    handle, _, _, _ = golden_world
    sources = (GoldenSource * 4)(GoldenSource(8., 1, 1), GoldenSource(3., 2, 0),
                                  GoldenSource(-0., 1, 1), GoldenSource(9., 0, 1))
    assert copied_plan(plan(library, handle, sources)) == ([0, 2], [])
    empty = GoldenView()
    assert library.tiemu_world_golden_plan(handle, None, 0, -(1 << 63), ct.byref(empty)) == 0
    assert copied_plan(empty) == ([], [])


@pytest.mark.parametrize('invalid', ['null_handle', 'null_sources', 'overflow', 'negative_trait',
                                    'invalid_trait', 'boss', 'nan', 'inf', 'null_output'])
def test_invalid_golden_plan_preserves_previous_plan_output_and_numeric_access(golden_abi, golden_world, invalid):
    _, library = golden_abi
    handle, live_sources, _, _ = golden_world
    for prior_plan in (False, True):
        assert library.tiemu_world_sync(handle, live_sources, 2) == 0
        original = sources_for(31)
        previous = plan(library, handle, original) if prior_plan else None
        previous_values = copied_plan(previous) if previous is not None else None
        sources = sources_for(31)
        output = GoldenView(reset_count=999, selected_count=888)
        before = bytes_of(output)
        arguments = [handle, sources, len(sources), 20, ct.byref(output)]
        if invalid == 'null_handle':
            arguments[0] = None
        elif invalid == 'null_sources':
            arguments[1] = None
        elif invalid == 'overflow':
            arguments[2] = ct.c_size_t(-1).value
        elif invalid == 'negative_trait':
            sources[2].trait = -1
        elif invalid == 'invalid_trait':
            sources[2].trait = 3
        elif invalid == 'boss':
            sources[2].boss = 2
        elif invalid == 'nan':
            sources[2].score = float('nan')
        elif invalid == 'inf':
            sources[2].score = float('inf')
        elif invalid == 'null_output':
            arguments[4] = None
        assert library.tiemu_world_golden_plan(*arguments) == 1
        assert bytes_of(output) == before
        if previous is not None:
            assert copied_plan(previous) == previous_values
        assert library.tiemu_world_get_view(handle, ct.byref(_View())) == (3 if prior_plan else 0)


def test_golden_plan_accepts_membership_dirty_but_rejects_failed_encounters(golden_abi, golden_world):
    _, library = golden_abi
    handle, live_sources, _, ids = golden_world
    assigned = np.empty(2, dtype=np.uint64)
    assert library.tiemu_world_reconcile(handle, ids.ctypes.data, 2, None, 0, assigned.ctypes.data, 2) == 0
    sources = sources_for(31)
    assert copied_plan(plan(library, handle, sources)) == expected_plan(sources, 20)
    assert library.tiemu_world_sync(handle, live_sources, 2) == 0
    # A nonnull config enters the encounter kernel and poisons the World on its
    # missing required inputs, rather than failing only the World argument guard.
    state = np.zeros(625, dtype=np.uint32)
    state[-1] = 624
    config, result = _Config(), _Result()
    config.entities = 2
    assert library.tiemu_world_run_encounters(handle, ct.byref(config), state.ctypes.data,
                                              state.ctypes.data, 625, ct.byref(result)) == 1
    output = GoldenView(reset_count=999, selected_count=888)
    before = bytes_of(output)
    assert library.tiemu_world_golden_plan(handle, sources, len(sources), 20, ct.byref(output)) == 3
    assert bytes_of(output) == before
    assert library.tiemu_world_sync(handle, live_sources, 2) == 0
    assert copied_plan(plan(library, handle, sources)) == expected_plan(sources, 20)


@pytest.mark.parametrize('invalid', [False, True])
def test_only_successful_golden_plan_invalidates_prepared_scoring_token(golden_abi, golden_world, invalid):
    native, library = golden_abi
    handle, _, _, _ = golden_world
    matrix, prepared = np.eye(12), _ScoringView()
    assert library.tiemu_world_scoring_prepare(handle, None, 0, matrix.ctypes.data, 144,
                                              ct.byref(prepared)) == 0
    evaluator = native.BatchEvaluator()
    try:
        sources, output = sources_for(5), GoldenView()
        if invalid:
            sources[0].boss = 2
        assert library.tiemu_world_golden_plan(handle, sources, 5, 20, ct.byref(output)) == (1 if invalid else 0)
        result = _ScoringView(rows=999, token=123)
        before = bytes_of(result)
        assert library.tiemu_world_scoring_finish(handle, prepared.token, evaluator._live(),
            None, 0, None, 0, ct.byref(result)) == (0 if invalid else 3)
        if not invalid:
            assert bytes_of(result) == before
    finally:
        evaluator.close()


def test_copied_golden_positions_survive_reuse_and_world_close(golden_abi, golden_world):
    _, library = golden_abi
    handle, _, _, _ = golden_world
    sources = sources_for(31)
    owned = copied_plan(plan(library, handle, sources))
    expected = expected_plan(sources, 20)
    for row in sources:
        row.score = -row.score
        row.trait = 2
    assert copied_plan(plan(library, handle, sources)) == ([], [])
    assert library.tiemu_world_destroy(ct.byref(handle)) == 0
    assert owned == expected
    output = GoldenView(reset_count=999)
    before = bytes_of(output)
    assert library.tiemu_world_golden_plan(handle, sources, len(sources), 20, ct.byref(output)) == 1
    assert bytes_of(output) == before
