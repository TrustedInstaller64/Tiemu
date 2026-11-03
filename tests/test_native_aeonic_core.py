"""Cloud-only ordered candidate pool contracts for the C ABI."""
import ctypes as ct

import numpy as np
import pytest

from native_backend import configure_backend


@pytest.fixture
def api():
    native = configure_backend('cpp')
    library = native._lib
    signatures = {
        'create': [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int64, ct.c_void_p],
        'pool_size': [ct.c_void_p, ct.c_uint32, ct.c_int64, ct.c_void_p],
        'pool_at': [ct.c_void_p, ct.c_uint32, ct.c_int64, ct.c_size_t, ct.c_void_p],
        'update': [ct.c_void_p, ct.c_size_t, ct.c_uint8, ct.c_uint8],
        'destroy': [ct.c_void_p],
    }
    try:
        for name, args in signatures.items():
            assert hasattr(library, 'tiemu_aeonic_' + name), f'Missing aeonic index C ABI: {name}'
            function = getattr(library, 'tiemu_aeonic_' + name)
            function.argtypes, function.restype = args, ct.c_int
        yield library
    finally:
        configure_backend('python')


def create(api, golden, titans, holders, reincarnator):
    arrays = [np.array(value, dtype=np.uint8) for value in (golden, titans, holders)]
    handle = ct.c_void_p()
    assert api.tiemu_aeonic_create(*(a.ctypes.data for a in arrays), len(golden), reincarnator,
                                  ct.byref(handle)) == 0
    # The caller's flags are never retained.
    for array in arrays:
        array[:] = 0
    return handle


def pool(api, handle, kind, excluded=-1):
    count = ct.c_size_t(999)
    assert api.tiemu_aeonic_pool_size(handle, kind, excluded, ct.byref(count)) == 0
    result = []
    for rank in range(count.value):
        position = ct.c_size_t(999)
        assert api.tiemu_aeonic_pool_at(handle, kind, excluded, rank, ct.byref(position)) == 0
        result.append(position.value)
    # A failed lookup cannot overwrite the caller's output sentinel.
    sentinel = ct.c_size_t(999)
    assert api.tiemu_aeonic_pool_at(handle, kind, excluded, count.value, ct.byref(sentinel)) == 1
    assert sentinel.value == 999
    return result


@pytest.mark.parametrize('reincarnator', [-1, 0, 4])
@pytest.mark.parametrize('golden', [[0]*6, [1, 0, 0, 1, 1, 0]])
def test_every_pool_retains_population_order_duplicate_reincarnator_and_updates(api, reincarnator, golden):
    titans, holders = [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0]
    handle = create(api, golden, titans, holders, reincarnator)
    try:
        for update in [None, (3, 0, 0), (0, 1, 1), (1, 0, 1), (5, 0, 0)]:
            if update:
                index, titan, holder = update
                assert api.tiemu_aeonic_update(handle, index, titan, holder) == 0
                titans[index], holders[index] = titan, holder
            attackers = ([reincarnator] if reincarnator >= 0 else []) + [i for i, g in enumerate(golden) if g]
            attackers = attackers or [i for i, t in enumerate(titans) if t] or list(range(6))
            assert pool(api, handle, 0) == attackers
            for excluded in range(-1, 6):
                assert pool(api, handle, 1, excluded) == [i for i in range(6) if i != excluded]
                assert pool(api, handle, 2, excluded) == [i for i in range(6) if i != excluded and (titans[i] or holders[i])]
                assert pool(api, handle, 3, excluded) == [i for i in range(6) if i != excluded and titans[i]]
        before = pool(api, handle, 2)
        for args in [(6, 0, 0), (0, 2, 0), (0, 0, 2)]:
            assert api.tiemu_aeonic_update(handle, *args) == 1
            assert pool(api, handle, 2) == before
        for kind, excluded in [(4, -1), (1, 6), (1, -2), (0, 0)]:
            out = ct.c_size_t(77)
            assert api.tiemu_aeonic_pool_size(handle, kind, excluded, ct.byref(out)) == 1
            assert out.value == 77
    finally:
        assert api.tiemu_aeonic_destroy(ct.byref(handle)) == 0
        assert not handle.value
        assert api.tiemu_aeonic_destroy(ct.byref(handle)) == 0


def test_empty_pool_and_strict_creation_validation(api):
    handle = ct.c_void_p()
    assert api.tiemu_aeonic_create(None, None, None, 0, -1, ct.byref(handle)) == 0
    try:
        for kind in range(4):
            assert pool(api, handle, kind) == []
        assert api.tiemu_aeonic_create(None, None, None, 0, -1, ct.byref(handle)) == 1
    finally:
        assert api.tiemu_aeonic_destroy(ct.byref(handle)) == 0
    flags = np.array([0, 1, 0], dtype=np.uint8)
    pointer = flags.ctypes.data
    for values in [(None, pointer, pointer, 3, -1), (pointer, None, pointer, 3, -1),
                   (pointer, pointer, None, 3, -1), (pointer, pointer, pointer, 3, 3),
                   (pointer, pointer, pointer, 3, -2),
                   (pointer, pointer, pointer, ct.c_size_t(-1).value, -1)]:
        assert api.tiemu_aeonic_create(*values, ct.byref(handle)) == 1
        assert not handle.value
    flags[0] = 2
    assert api.tiemu_aeonic_create(pointer, pointer, pointer, 3, -1, ct.byref(handle)) == 1
    assert not handle.value
