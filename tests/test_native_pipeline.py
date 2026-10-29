"""Ordered fusion and row-parallel contracts against the existing scalar kernels.

These tests deliberately use the already differential-tested scalar entry points
as the same-platform reference. Worker count must never change a row's arithmetic
or allow one request's buffers to be used by another request.
"""
import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


@pytest.fixture(scope="module")
def native():
    return importlib.import_module("_tiemu_native")


def scalar_pipeline(native, values, factor, tide, limit):
    # Pathstrider skips purification without touching its array for these cases.
    purified = values if factor <= 0 or values.size < 2 else native.purify(values, factor)
    return native.normalize(np.clip(purified + tide, 0, None), limit)


@pytest.mark.parametrize("factor", [-1.0, 0.0, 0.1, float("nan")])
@pytest.mark.parametrize("values,tide,limit", [
    ([4., 4., 1.], [-1., .5, -8.], 2.),
    ([0., 0., 0.], [0., 0., 0.], 3.),
    ([2.], [-3.], 3.),
    ([], [], 3.),
    ([np.inf, 1., -1.], [1., 0., 1.], 3.),
    ([np.nan, 1., 2.], [0., -1., 0.], 3.),
    ([1e200, 1e200, 1.], [0., 0., 0.], 3.),
])
def test_fused_pipeline_keeps_purification_tide_clip_and_cap_order(native, factor, values, tide, limit):
    assert hasattr(native, "normalize_pipeline"), "ordered native pipeline has not been implemented"
    values, tide = np.array(values), np.array(tide)
    originals = values.copy(), tide.copy()
    with np.errstate(all="ignore"):
        expected = scalar_pipeline(native, values, factor, tide, limit)
        actual = native.normalize_pipeline(values, factor, tide, limit)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, originals[0])
    np.testing.assert_array_equal(tide, originals[1])


def test_fused_pipeline_accepts_readonly_strided_inputs_without_aliasing(native):
    values = np.arange(24, dtype=np.float64)[::2]
    tide = np.linspace(-2, 2, 24)[::2]
    values.flags.writeable = tide.flags.writeable = False
    actual = native.normalize_pipeline(values, .05, tide, 5.)
    np.testing.assert_array_equal(actual, scalar_pipeline(native, values, .05, tide, 5.))
    assert not np.shares_memory(actual, values)
    assert not np.shares_memory(actual, tide)


def batch_inputs(rows, seed=731):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-2, 8, (rows, 12))
    matrix = rng.uniform(-.5, 1., (12, 12))
    multipliers = rng.uniform(.9, 1.2, rows)
    distribution = rng.uniform(0, 1, 12)
    distribution /= distribution.sum()
    bosses = (np.arange(rows) % 3 == 0).astype(np.int64)
    return values, matrix, multipliers, distribution, bosses


def scalar_concepts(native, values, matrix, multipliers, distribution, bosses):
    paths = np.empty((len(values), matrix.shape[1]))
    stats = np.empty((len(values), 4))
    for i, value in enumerate(values):
        # Match Pathstrider.recalculate_concepts: sanitize the entire titan row.
        if not np.all(np.isfinite(value)):
            value = np.ones(len(value))
        paths[i] = native.path_affinities(value, matrix)
        stats[i] = native.concepts(paths[i], multipliers[i], distribution, bool(bosses[i]))
    return paths, stats


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("rows", [0, 1, 7, 8, 37])
@pytest.mark.parametrize("with_distribution", [False, True])
def test_batch_rows_match_scalar_kernels_on_both_sides_of_thread_threshold(native, workers, rows, with_distribution):
    assert hasattr(native, "BatchEvaluator"), "native batch evaluator has not been implemented"
    inputs = list(batch_inputs(rows))
    if not with_distribution:
        inputs[3] = None
    originals = [value.copy() if value is not None else None for value in inputs]
    expected = scalar_concepts(native, *inputs)
    evaluator = native.BatchEvaluator(workers=workers, min_rows=8)
    for _ in range(3):
        actual = evaluator.concepts(*inputs)
        assert actual[0].shape == (rows, 12)
        assert actual[1].shape == (rows, 4)
        for found, reference in zip(actual, expected):
            np.testing.assert_array_equal(found, reference)
    for found, before in zip(inputs, originals):
        if found is not None:
            np.testing.assert_array_equal(found, before)


def test_batch_nonfinite_titan_rows_are_sanitized_and_ties_keep_first_path(native):
    values = np.array([[np.nan] + [2.] * 11, [np.inf] + [0.] * 11,
                       [-np.inf] + [1.] * 11, [0.] * 12, [1., 1.] + [0.] * 10])
    inputs = (values, np.eye(12), np.ones(5), np.linspace(.01, .8, 12), np.array([0, 1, 0, 0, 1]))
    expected = scalar_concepts(native, *inputs)
    actual = native.BatchEvaluator(workers=4, min_rows=1).concepts(*inputs)
    for found, reference in zip(actual, expected):
        np.testing.assert_array_equal(found, reference)
    np.testing.assert_array_equal(values[0], [np.nan] + [2.] * 11)


def test_batch_accepts_noncontiguous_inputs_and_preserves_results_after_reuse(native):
    values, matrix, multipliers, distribution, bosses = batch_inputs(38)
    inputs = (values[::2], matrix[:, ::2], multipliers[::2], distribution[::2], bosses[::2])
    for value in inputs:
        value.flags.writeable = False
    evaluator = native.BatchEvaluator(workers=4, min_rows=1)
    actual = evaluator.concepts(*inputs)
    expected = scalar_concepts(native, *inputs)
    evaluator.concepts(*batch_inputs(3, seed=19))
    for found, reference in zip(actual, expected):
        np.testing.assert_array_equal(found, reference)


def test_shared_batch_evaluator_serializes_requests_without_mixing_buffers(native):
    evaluator = native.BatchEvaluator(workers=4, min_rows=1)
    inputs = [batch_inputs(41, seed=seed) for seed in range(6)]
    expected = [scalar_concepts(native, *arguments) for arguments in inputs]
    with ThreadPoolExecutor(max_workers=2) as callers:
        actual = list(callers.map(lambda arguments: evaluator.concepts(*arguments), inputs))
    for result, reference in zip(actual, expected):
        for found, wanted in zip(result, reference):
            np.testing.assert_array_equal(found, wanted)


def test_batch_close_is_idempotent_and_rejects_further_work(native):
    evaluator = native.BatchEvaluator(workers=4, min_rows=1)
    evaluator.concepts(*batch_inputs(19))
    evaluator.close()
    evaluator.close()
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.concepts(*batch_inputs(1))


@pytest.mark.parametrize("workers,min_rows", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_batch_rejects_invalid_worker_and_threshold_settings(native, workers, min_rows):
    with pytest.raises(ValueError):
        native.BatchEvaluator(workers=workers, min_rows=min_rows)


@pytest.mark.parametrize("operation", [
    lambda n: n.normalize_pipeline(np.ones((1, 12)), .1, np.ones(12), 5.),
    lambda n: n.normalize_pipeline(np.ones(12), .1, np.ones((1, 12)), 5.),
    lambda n: n.normalize_pipeline(np.ones(12), .1, np.ones(11), 5.),
    lambda n: n.BatchEvaluator().concepts(np.ones(12), np.eye(12), np.ones(1), None, np.zeros(1, dtype=np.int64)),
    lambda n: n.BatchEvaluator().concepts(np.ones((1, 12)), np.eye(11), np.ones(1), None, np.zeros(1, dtype=np.int64)),
    lambda n: n.BatchEvaluator().concepts(np.ones((1, 12)), np.empty((12, 0)), np.ones(1), None, np.zeros(1, dtype=np.int64)),
    lambda n: n.BatchEvaluator().concepts(np.ones((1, 12)), np.eye(12), np.ones(2), None, np.zeros(1, dtype=np.int64)),
    lambda n: n.BatchEvaluator().concepts(np.ones((1, 12)), np.eye(12), np.ones(1), np.ones(11), np.zeros(1, dtype=np.int64)),
    lambda n: n.BatchEvaluator().concepts(np.ones((1, 12)), np.eye(12), np.ones(1), None, np.zeros(2, dtype=np.int64)),
])
def test_fused_operations_reject_bad_shapes_before_native_memory_access(native, operation):
    with pytest.raises(ValueError):
        operation(native)


def test_vote_templates_keep_signed_nonfinite_weights_and_entity_accumulation_order(native):
    # Non-associative weights expose an accidental group-by-path reduction.
    dominant = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    logits = np.array([[1., -.7], [.2, 1.]])
    for weights in (np.array([1e16, 1., -1e16, -1., 3., 2.]),
                    np.array([np.inf, 1., -np.inf, 2., 0., -0.])):
        expected = np.zeros(2)
        with np.errstate(all="ignore"):
            for index, weight in zip(dominant, weights):
                # A one-entity native vote fixes exp rounding on this platform;
                # summing the contributions here catches grouped reductions.
                proposal = native.vote_totals(np.array([index]), np.ones(1), logits)
                expected += proposal * weight
        np.testing.assert_array_equal(native.vote_totals(dominant, weights, logits), expected)
