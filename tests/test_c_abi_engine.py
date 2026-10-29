"""Migration contracts: run only on the remote hosted CI while local load is forbidden."""
import importlib
import random

import numpy as np
import pytest


def native():
    return importlib.import_module("_tiemu_ffi")


def test_native_core_uses_c_abi_without_a_python_extension_binding():
    module = native()
    assert module.__binding__ == "c-abi"
    assert module.ABI_VERSION == 1
    assert module.__file__.endswith(".py")


def test_backend_ignores_old_inplace_extension_name(monkeypatch):
    import native_backend
    from types import SimpleNamespace
    current = native()
    old = SimpleNamespace(__file__="_tiemu_native.cpython-312.so")
    real_import = importlib.import_module
    def upgrade_environment(name):
        if name == "_tiemu_native":
            return old
        if name == "_tiemu_ffi":
            return current
        return real_import(name)
    monkeypatch.setattr(native_backend.importlib, "import_module", upgrade_environment)
    try:
        assert native_backend.configure_backend("cpp") is current
    finally:
        native_backend.configure_backend("python")


def original_grid(activity, stability, rng, steps, grid):
    a, s = activity.copy(), stability.copy()
    for _ in range(steps):
        for i in range(grid):
            for j in range(grid):
                index = i * grid + j
                total_a = total_s = 0.0
                count = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if (di or dj) and 0 <= i + di < grid and 0 <= j + dj < grid:
                            neighbor = (i + di) * grid + j + dj
                            total_a += a[neighbor]
                            total_s += s[neighbor]
                            count += 1
                mean_a = total_a / count if count else 0.0
                mean_s = total_s / count if count else 0.0
                change_a = (mean_a - a[index]) * 0.1 + rng.uniform(-0.1, 0.1)
                change_s = (mean_s - s[index]) * 0.1 + rng.uniform(-0.1, 0.1)
                a[index] = max(0.0, a[index] + change_a)
                s[index] = max(0.0, s[index] + change_s)
    return a, s


@pytest.mark.parametrize("seed", [0, 731, 2**65 + 17])
@pytest.mark.parametrize("grid,steps", [(1, 0), (1, 7), (3, 4), (10, 2)])
def test_whole_inorganic_phase_preserves_python_random_state_and_order(seed, grid, steps):
    rng = random.Random(seed)
    rng.gauss(0, 1)  # Keep a populated cached Gaussian across the uniform stream.
    before = rng.getstate()
    a = np.linspace(1, 5, grid * grid)
    s = np.linspace(5, 1, grid * grid)
    mt = np.array(before[1], dtype=np.uint32)
    a_before, s_before, mt_before = a.copy(), s.copy(), mt.copy()
    expected_a, expected_s = original_grid(a, s, rng, steps, grid)
    actual_a, actual_s, actual_mt = native().inorganic_run(a, s, mt, steps, grid)
    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_s, expected_s)
    assert tuple(map(int, actual_mt)) == rng.getstate()[1]
    restored = random.Random()
    restored.setstate((before[0], tuple(map(int, actual_mt)), before[2]))
    assert restored.getstate() == rng.getstate()
    assert restored.gauss(0, 1) == rng.gauss(0, 1)
    assert restored.random() == rng.random()
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(s, s_before)
    np.testing.assert_array_equal(mt, mt_before)


def test_inorganic_chunk_boundaries_preserve_arrays_and_rng():
    module = native()
    mt = np.array(random.Random(731).getstate()[1], dtype=np.uint32)
    a, s = np.arange(1., 10.), np.arange(9., 0., -1.)
    full = module.inorganic_run(a, s, mt, 257, 3)
    current = (a, s, mt)
    for steps in (1, 100, 0, 99, 57):
        current = module.inorganic_run(*current, steps, 3)
    for actual, expected in zip(current, full):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", ["negative_steps", "wrong_grid", "short_state", "invalid_index"])
def test_inorganic_engine_rejects_invalid_contracts(case):
    module = native()
    method = module.inorganic_run
    a, s = np.ones(9), np.ones(9)
    mt = np.array(random.Random(1).getstate()[1], dtype=np.uint32)
    steps, grid = 1, 3
    if case == "negative_steps":
        steps = -1
    elif case == "wrong_grid":
        grid = 2
    elif case == "short_state":
        mt = mt[:-1]
    else:
        mt[-1] = 625
    with pytest.raises(ValueError):
        method(a, s, mt, steps, grid)
