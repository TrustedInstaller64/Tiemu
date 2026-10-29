"""Application contracts for ordered batch scoring and fused normalization.

Run on remote CI: this file must not be executed on the development Mac while
the user's local compute restriction is active.
"""
import copy
import importlib
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from constants import PATH_NAMES
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import configure_backend
from parliament_manager import ParliamentManager
from population_manager import PopulationManager


@pytest.fixture
def native():
    module = configure_backend("cpp")
    try:
        yield module
    finally:
        configure_backend("python")


def evaluator_module():
    try:
        return importlib.import_module("native_evaluation")
    except ImportError as exc:
        pytest.fail(f"Ordered entity evaluation has not been implemented: {exc}")


def parliament():
    manager = ParliamentManager()
    manager.seats = {name: index + 2 for index, name in enumerate(PATH_NAMES)}
    manager.suppression_factor = 0.7
    return manager


def entities():
    model = TitanToPathModel()
    population = [Pathstrider(str(index), np.arange(1., 13.) + index,
                             titan_to_path_model=model) for index in range(7)]
    # The old cache intentionally disagrees with current titan values. The
    # multiplier must use that old cache before scoring refreshes the paths.
    for index, entity in enumerate(population):
        if index % 2 == 0:
            entity._path_affinities = np.roll(np.arange(1., 13.) / 78., index)
        entity.titan_affinities = entity.titan_affinities[::-1].copy()
    population[1].titan_affinities[:] = 0
    population[2].titan_affinities[3] = np.nan
    population[3].is_titan_boss = True
    population[4].titan_affinities[5] = np.inf
    return population


def score_scalar(population, manager, distribution):
    for entity in population:
        multiplier = manager.get_zeitgeist_multiplier(entity.path_affinities)
        entity.recalculate_concepts(multiplier, distribution)


def assert_entities_equal(actual, expected):
    assert [entity.name for entity in actual] == [entity.name for entity in expected]
    for current, reference in zip(actual, expected):
        np.testing.assert_array_equal(current.titan_affinities, reference.titan_affinities)
        np.testing.assert_array_equal(current.path_affinities, reference.path_affinities)
        np.testing.assert_array_equal(
            [current.activity, current.stability, current.score, current.heroic_tendency],
            [reference.activity, reference.stability, reference.score, reference.heroic_tendency])


def rng_state():
    return (pickle.dumps((random.getstate(), np.random.get_state())),
            torch.get_rng_state().numpy().tobytes())


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("distribution", [None, np.arange(1., 13.) / 78.])
def test_batch_keeps_old_path_multiplier_repairs_nonfinite_and_preserves_rng(native, workers, distribution):
    evaluate = evaluator_module().evaluate_entities
    actual, manager = entities(), parliament()
    expected = copy.deepcopy(actual)
    before_rng = rng_state()
    with np.errstate(all="ignore"):
        score_scalar(expected, manager, distribution)
        evaluator = native.BatchEvaluator(workers=workers, min_rows=1)
        try:
            evaluate(iter(actual), manager, distribution, evaluator=evaluator)
        finally:
            evaluator.close()
    assert_entities_equal(actual, expected)
    assert rng_state() == before_rng


def test_batch_outputs_remain_valid_after_next_call_and_evaluator_close(native):
    evaluate = evaluator_module().evaluate_entities
    first, second, manager = entities(), entities(), parliament()
    evaluator = native.BatchEvaluator(workers=2, min_rows=1)
    with np.errstate(all="ignore"):
        evaluate(first, manager, None, evaluator=evaluator)
        expected = copy.deepcopy(first)
        for entity in second:
            entity.titan_affinities *= 3
        evaluate(second, manager, None, evaluator=evaluator)
        evaluator.close()
    assert_entities_equal(first, expected)


def test_empty_batch_does_not_touch_closed_evaluator(native):
    evaluator = native.BatchEvaluator(workers=2, min_rows=1)
    evaluator.close()
    evaluator_module().evaluate_entities(iter(()), parliament(), None, evaluator=evaluator)


def test_simulation_close_releases_native_workers_and_console(native):
    from simulation import AeonEvolution
    simulation = object.__new__(AeonEvolution)
    simulation._native_evaluator = native.BatchEvaluator(workers=2, min_rows=1)
    evaluator = simulation._native_evaluator
    closed = []
    simulation.debugger = SimpleNamespace(close=lambda: closed.append(True))
    simulation.close()
    simulation.close()
    assert closed
    with pytest.raises(RuntimeError):
        evaluator.concepts(np.ones((1, 12)), np.eye(12), np.ones(1), None,
                           np.zeros(1, dtype=np.int64))


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_missing_evaluator_uses_existing_scalar_semantics(backend):
    configure_backend(backend)
    try:
        actual, manager = entities(), parliament()
        expected = copy.deepcopy(actual)
        with np.errstate(all="ignore"):
            score_scalar(expected, manager, None)
            evaluator_module().evaluate_entities(actual, manager, None)
        assert_entities_equal(actual, expected)
    finally:
        configure_backend("python")


@pytest.mark.parametrize("case", ["duplicate", "different_mapping"])
def test_dependent_or_heterogeneous_entities_keep_scalar_order(native, case):
    actual, manager = entities(), parliament()
    if case == "duplicate":
        actual = [actual[0], actual[0]]
    else:
        actual[1]._titan_to_path_model = copy.deepcopy(actual[1]._titan_to_path_model)
        actual[1]._titan_to_path_model.titan_to_path_matrix *= -1
    expected = copy.deepcopy(actual)
    evaluator = native.BatchEvaluator(workers=2, min_rows=1)
    with np.errstate(all="ignore"):
        score_scalar(expected, manager, None)
        try:
            evaluator_module().evaluate_entities(actual, manager, None, evaluator=evaluator)
        finally:
            evaluator.close()
    assert_entities_equal(actual, expected)


def expected_multiplier(manager, paths):
    if np.sum(list(manager.seats.values())) == 0:
        return 1.0
    normalized = np.array([manager.seats[name] for name in PATH_NAMES]) / manager.total_seats
    alignment = np.dot(paths, normalized) - np.dot(1 - paths, normalized)
    return max(0.9, min(1.2, 1.05 + 0.15 * np.tanh(alignment * 3) * manager.suppression_factor))


def test_seat_cache_observes_public_dictionary_edits_replacement_total_and_suppression():
    manager = parliament()
    paths = np.arange(1., 13.) / 78.
    for change in (
        lambda: None,
        lambda: manager.seats.__setitem__(PATH_NAMES[0], 71),
        lambda: setattr(manager, "suppression_factor", -0.4),
        lambda: setattr(manager, "total_seats", 17),
        lambda: setattr(manager, "seats", {name: 0 for name in PATH_NAMES}),
        lambda: manager.seats.__setitem__("external", 4),
        lambda: manager.seats.pop("external"),
    ):
        change()
        assert manager.get_zeitgeist_multiplier(paths) == expected_multiplier(manager, paths)


@pytest.mark.parametrize("factor", [-0.2, 0., 0.1, np.nan])
@pytest.mark.parametrize("dimension", [1, 12])
def test_normalization_pipeline_keeps_cache_invalidation_and_existing_array_aliases(native, factor, dimension):
    model = TitanToPathModel()
    manager = PopulationManager(set(), {}, np.ones(12), 0.1, factor, 12, 0.1,
                                model, 2., 1., 100.)
    manager.cosmic_tide_vector = np.linspace(-0.4, 0.7, dimension)
    actual = Pathstrider("actual", np.arange(1., dimension + 1), titan_to_path_model=model)
    actual._path_affinities = np.arange(1., 13.) / 78.
    expected = copy.deepcopy(actual)
    actual_old_array, expected_old_array = actual.titan_affinities, expected.titan_affinities
    old_cache = actual._path_affinities
    with np.errstate(all="ignore"):
        expected.internal_purification(factor)
        expected.titan_affinities += manager.cosmic_tide_vector
        expected.titan_affinities = expected.titan_affinities.clip(min=0)
        manager.normalize_affinities(expected)
        manager.recalculate_and_normalize_entity(actual, None, None)
    np.testing.assert_array_equal(actual.titan_affinities, expected.titan_affinities)
    np.testing.assert_array_equal(actual_old_array, expected_old_array)
    if factor <= 0 or dimension < 2:
        assert actual._path_affinities is old_cache
    else:
        assert actual._path_affinities is None
