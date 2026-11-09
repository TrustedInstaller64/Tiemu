"""Ordered-grid contracts for the optional neighbor-table candidate.

Run against both the default and macro-enabled native libraries.
The oracle freezes the original shallow-list-copy loop, independently of either
C++ implementation. It deliberately retains per-neighbor addition order.
"""
import importlib
import random

import numpy as np
import pytest


@pytest.fixture(scope="module")
def native():
    return importlib.import_module("_tiemu_ffi")


def frozen_grid(activity, stability, rng, steps, grid):
    cells = [dict(activity=float(a), stability=float(s))
             for a, s in zip(activity, stability)]
    for _ in range(steps):
        next_cells = list(cells)
        for i in range(grid):
            for j in range(grid):
                index = i * grid + j
                current = cells[index]
                total_a = total_s = 0.0
                count = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid and 0 <= nj < grid:
                            neighbor = cells[ni * grid + nj]
                            total_a += neighbor["activity"]
                            total_s += neighbor["stability"]
                            count += 1
                mean_a = total_a / count if count else 0.0
                mean_s = total_s / count if count else 0.0
                change_a = (mean_a - current["activity"]) * 0.1 + rng.uniform(-0.1, 0.1)
                change_s = (mean_s - current["stability"]) * 0.1 + rng.uniform(-0.1, 0.1)
                next_cells[index]["activity"] = max(0.0, current["activity"] + change_a)
                next_cells[index]["stability"] = max(0.0, current["stability"] + change_s)
        cells = next_cells
    return tuple(np.array([cell[field] for cell in cells], dtype=np.float64)
                 for field in ("activity", "stability"))


def inputs(grid):
    # Corners/edges/interiors differ, and mixed magnitudes expose changed
    # addition order; a spatially uniform fixture would hide these defects.
    pattern = np.array([0.0, 1e-12, 1e16, 3.25, 1e-3, 1e15, 7.5, -1.0])
    activity = np.resize(pattern, grid * grid)
    stability = np.resize(pattern[::-1], grid * grid)
    return activity, stability


def assert_same_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mistake", ["reverse_neighbors", "simultaneous_update"])
def test_reference_fixture_distinguishes_wrong_neighbor_algorithms(native, mistake):
    # Negative controls are confined to this test. For cell 0, (1e16 + 1) + 1
    # and (1 + 1) + 1e16 differ; later cells also observe earlier cell updates.
    activity = np.array([0.0, 1e16, 1.0, 1.0])
    stability = np.array([1.0, 1.0, 1e16, 0.0])
    reference_rng = random.Random(19)
    mt = np.array(reference_rng.getstate()[1], dtype=np.uint32)
    expected = frozen_grid(activity, stability, reference_rng, 1, 2)
    actual = native.inorganic_run(activity, stability, mt, 1, 2)
    assert_same_bits(actual[0], expected[0])
    assert_same_bits(actual[1], expected[1])
    assert tuple(map(int, actual[2])) == reference_rng.getstate()[1]
    wrong_a, wrong_s = activity.copy(), stability.copy()
    rng = random.Random(19)
    for i in range(2):
        for j in range(2):
            index = 2 * i + j
            neighbors = [2 * (i + di) + j + dj
                         for di in (-1, 0, 1) for dj in (-1, 0, 1)
                         if (di or dj) and 0 <= i + di < 2 and 0 <= j + dj < 2]
            if mistake == "reverse_neighbors":
                neighbors.reverse()
            source_a, source_s = ((activity, stability) if mistake == "simultaneous_update"
                                  else (wrong_a, wrong_s))
            total_a = total_s = 0.0
            for n in neighbors:
                total_a += float(source_a[n])
                total_s += float(source_s[n])
            mean_a, mean_s = total_a / len(neighbors), total_s / len(neighbors)
            current_a, current_s = float(wrong_a[index]), float(wrong_s[index])
            change_a = (mean_a - current_a) * 0.1 + rng.uniform(-0.1, 0.1)
            change_s = (mean_s - current_s) * 0.1 + rng.uniform(-0.1, 0.1)
            wrong_a[index] = max(0.0, current_a + change_a)
            wrong_s[index] = max(0.0, current_s + change_s)
    assert any(actual.tobytes() != correct.tobytes()
               for actual, correct in zip((wrong_a, wrong_s), expected))


@pytest.mark.parametrize("seed", [0, 2**65 + 17])
@pytest.mark.parametrize("grid,steps", [
    (1, 0), (1, 7), (2, 1), (2, 100), (3, 1), (3, 7),
    (10, 0), (10, 1), (10, 100), (17, 1), (17, 9),
])
def test_neighbor_table_preserves_sequential_grid_and_full_rng(native, seed, grid, steps):
    # Catches wraparound boundaries, self-neighbors, simultaneous updates,
    # reordered sums, and any changed number/order of per-cell RNG draws.
    rng = random.Random(seed)
    rng.gauss(0.0, 1.0)
    for _ in range(307):
        rng.random()
    before = rng.getstate()
    activity, stability = inputs(grid)
    mt = np.array(before[1], dtype=np.uint32)
    originals = activity.copy(), stability.copy(), mt.copy()
    expected = frozen_grid(activity, stability, rng, steps, grid)
    actual_a, actual_s, actual_mt = native.inorganic_run(activity, stability, mt, steps, grid)
    assert_same_bits(actual_a, expected[0])
    assert_same_bits(actual_s, expected[1])
    assert tuple(map(int, actual_mt)) == rng.getstate()[1]
    restored = random.Random()
    restored.setstate((before[0], tuple(map(int, actual_mt)), before[2]))
    assert restored.getstate() == rng.getstate()
    assert restored.gauss(0.0, 1.0) == rng.gauss(0.0, 1.0)
    assert restored.random() == rng.random()
    for value, original in zip((activity, stability, mt), originals):
        assert_same_bits(value, original)


@pytest.mark.parametrize("grid", [1, 2, 10])
def test_neighbor_table_chunk_rebuilds_preserve_state(native, grid):
    # Production headless runs use <=100-generation segments; rebuilding a
    # table must neither reset the world nor consume an extra random number.
    activity, stability = inputs(grid)
    rng = random.Random(731)
    mt = np.array(rng.getstate()[1], dtype=np.uint32)
    expected_a, expected_s = frozen_grid(activity, stability, rng, 203, grid)
    actual = activity, stability, mt
    for steps in (1, 100, 0, 100, 2):
        actual = native.inorganic_run(*actual, steps, grid)
    assert_same_bits(actual[0], expected_a)
    assert_same_bits(actual[1], expected_s)
    assert tuple(map(int, actual[2])) == rng.getstate()[1]


def test_zero_steps_preserve_nonfinite_and_signed_zero_bits(native):
    # A zero-step call is a copy/state round trip, not a sanitization pass.
    activity = np.array([-0.0, np.inf, -np.inf, np.nan])
    stability = np.array([np.nan, -0.0, np.inf, -np.inf])
    mt = np.array(random.Random(41).getstate()[1], dtype=np.uint32)
    actual = native.inorganic_run(activity, stability, mt, 0, 2)
    for value, expected in zip(actual, (activity, stability, mt)):
        assert_same_bits(value, expected)


@pytest.mark.parametrize("case", [
    "negative_steps", "zero_grid", "negative_grid", "wrong_grid", "short_state", "invalid_index",
])
def test_neighbor_candidate_keeps_invalid_input_contract(native, case):
    activity, stability = inputs(2)
    mt = np.array(random.Random(5).getstate()[1], dtype=np.uint32)
    steps, grid = 1, 2
    if case == "negative_steps":
        steps = -1
    elif case == "zero_grid":
        grid = 0
    elif case == "negative_grid":
        grid = -1
    elif case == "wrong_grid":
        grid = 3
    elif case == "short_state":
        mt = mt[:-1]
    else:
        mt[-1] = 625
    originals = activity.copy(), stability.copy(), mt.copy()
    with pytest.raises(ValueError):
        native.inorganic_run(activity, stability, mt, steps, grid)
    for value, original in zip((activity, stability, mt), originals):
        assert_same_bits(value, original)
