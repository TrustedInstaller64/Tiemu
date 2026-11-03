"""C1 scoring contracts; execute only on hosted CI, never on the reserved Mac.

The initial RED commit allows only a missing experimental method to xfail.
Once the method exists, every assertion is mandatory. Local numerical tolerance
is not a world-trajectory equivalence guarantee.
"""
import copy
import importlib
import pickle
import random

import numpy as np
import pytest

from config import config
from native_backend import configure_backend
from test_native_evaluation import entities, parliament, score_scalar


_native_module = importlib.import_module("_tiemu_ffi")
pytestmark = pytest.mark.xfail(
    not hasattr(_native_module.BatchEvaluator, "concepts_zeitgeist"),
    raises=AttributeError, strict=True,
    reason="C1 RED: standalone fused zeitgeist scoring API is not implemented")


@pytest.fixture
def native(monkeypatch):
    module = configure_backend("cpp")
    try:
        # All tests, including opt-out cases, first prove that the new feature
        # exists. RED must not hide failures other than this missing method.
        getattr(module.BatchEvaluator, "concepts_zeitgeist")
        monkeypatch.setitem(config.setdefault("runtime", {}), "experimental_c_scoring", True)
        yield module
    finally:
        configure_backend("python")


def inputs(rows=37):
    rng = np.random.default_rng(712)
    values = rng.uniform(-2, 8, (rows, 12))
    matrix = rng.uniform(-.4, 1., (12, 12))
    old_paths = rng.uniform(0, 1, (rows, 12))
    old_paths /= old_paths.sum(axis=1, keepdims=True)
    seats = np.arange(1., 13.) / 100.
    distribution = np.arange(1., 13.) / 78.
    bosses = (np.arange(rows) % 3 == 0).astype(np.int64)
    return values, matrix, old_paths, seats, .7, distribution, bosses


def reference(native, args):
    values, matrix, old_paths, seats, suppression, distribution, bosses = args
    multipliers = []
    for old in old_paths:
        if seats is None:
            multipliers.append(1.)
        else:
            alignment = np.dot(old, seats) - np.dot(1 - old, seats)
            multipliers.append(max(.9, min(1.2, 1.05 + .15 * np.tanh(alignment * 3) * suppression)))
    evaluator = native.BatchEvaluator(1, 1)
    try:
        return evaluator.concepts(values, matrix, np.array(multipliers), distribution, bosses)
    finally:
        evaluator.close()


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("rows", [0, 1, 37])
def test_fused_scoring_matches_original_numpy_multiplier_formula(native, workers, rows):
    args = inputs(rows)
    before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    evaluator = native.BatchEvaluator(workers, 1)
    try:
        actual = evaluator.concepts_zeitgeist(*args)
    finally:
        evaluator.close()
    expected = reference(native, args)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-14, atol=2e-14)
    for value, previous in zip(args, before):
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, previous)


@pytest.mark.parametrize("seats,suppression", [(None, 100.), (np.zeros(12), 0.),
                                               (np.ones(12), -20.), (np.ones(12), 20.)])
def test_zero_seats_suppression_and_clamping(native, seats, suppression):
    args = list(inputs())
    args[3:5] = [seats, suppression]
    evaluator = native.BatchEvaluator(4, 1)
    try:
        actual = evaluator.concepts_zeitgeist(*args)
    finally:
        evaluator.close()
    for value, expected in zip(actual, reference(native, args)):
        np.testing.assert_allclose(value, expected, rtol=2e-14, atol=2e-14)


def test_one_and_four_workers_are_bitwise_identical_and_results_survive_close(native):
    args = inputs(129)
    serial, parallel = native.BatchEvaluator(1, 1), native.BatchEvaluator(4, 1)
    try:
        expected = serial.concepts_zeitgeist(*args)
        actual = parallel.concepts_zeitgeist(*args)
        parallel.concepts_zeitgeist(*inputs(7))
    finally:
        serial.close()
        parallel.close()
    for value, wanted in zip(actual, expected):
        np.testing.assert_array_equal(value, wanted)
    with pytest.raises(RuntimeError, match="closed"):
        parallel.concepts_zeitgeist(*args)


@pytest.mark.parametrize("index,bad", [(0, np.ones(12)), (1, np.eye(11)),
                                      (2, np.ones((37, 11))), (3, np.ones(11)),
                                      (5, np.ones(11)), (6, np.zeros(36, dtype=np.int64))])
def test_invalid_shapes_are_rejected_before_native_access(native, index, bad):
    args = list(inputs())
    args[index] = bad
    evaluator = native.BatchEvaluator(1, 1)
    try:
        with pytest.raises(ValueError):
            evaluator.concepts_zeitgeist(*args)
    finally:
        evaluator.close()


def test_c_abi_checks_output_capacity(native, monkeypatch):
    function = native._batch_concepts_zeitgeist
    def truncate_output_capacity(*args):
        args = list(args)
        args[-1] = 0
        return function(*args)
    monkeypatch.setattr(native, "_batch_concepts_zeitgeist", truncate_output_capacity)
    evaluator = native.BatchEvaluator(1, 1)
    try:
        with pytest.raises(ValueError):
            evaluator.concepts_zeitgeist(*inputs())
    finally:
        evaluator.close()


def test_application_keeps_old_cache_read_repairs_nonfinite_and_preserves_rng(native, monkeypatch):
    from native_evaluation import evaluate_entities
    actual, manager = entities(), parliament()
    expected = copy.deepcopy(actual)
    before_rng = pickle.dumps((random.getstate(), np.random.get_state()))
    calls = []
    original = native.BatchEvaluator.concepts_zeitgeist
    def observe(self, *args):
        calls.append(1)
        return original(self, *args)
    monkeypatch.setattr(native.BatchEvaluator, "concepts_zeitgeist", observe)
    evaluator = native.BatchEvaluator(4, 1)
    try:
        with np.errstate(all="ignore"):
            score_scalar(expected, manager, np.arange(1., 13.) / 78.)
            evaluate_entities(actual, manager, np.arange(1., 13.) / 78., evaluator=evaluator)
    finally:
        evaluator.close()
    assert calls == [1], "The intended experimental branch must actually execute"
    for current, previous in zip(actual, expected):
        np.testing.assert_array_equal(current.titan_affinities, previous.titan_affinities)
        np.testing.assert_array_equal(current.path_affinities, previous.path_affinities)
        np.testing.assert_allclose([current.score, current.activity, current.stability],
                                   [previous.score, previous.activity, previous.stability],
                                   rtol=2e-14, atol=2e-14)
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before_rng


@pytest.mark.parametrize("case", ["flag_off", "python_backend", "no_evaluator"])
def test_experiment_opt_out_keeps_reference_route(native, monkeypatch, case):
    from native_evaluation import evaluate_entities
    from test_native_evaluation import assert_entities_equal
    def forbidden(*args):
        raise AssertionError("C scoring must not execute when unavailable/disabled")
    monkeypatch.setattr(native.BatchEvaluator, "concepts_zeitgeist", forbidden)
    actual, manager = entities(), parliament()
    evaluator = native.BatchEvaluator(1, 1)
    if case == "flag_off":
        monkeypatch.setitem(config["runtime"], "experimental_c_scoring", False)
    elif case == "python_backend":
        configure_backend("python")
    expected = copy.deepcopy(actual)
    try:
        with np.errstate(all="ignore"):
            score_scalar(expected, manager, None)
            evaluate_entities(actual, manager, None,
                              evaluator=None if case == "no_evaluator" else evaluator)
    finally:
        evaluator.close()
    assert_entities_equal(actual, expected)
