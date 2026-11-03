"""Cloud-only application contracts for resident scoring preparation and caches."""
import functools
import importlib
import pickle
import random
import subprocess

import numpy as np
import pytest
import torch

from constants import PATH_NAMES
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import configure_backend
from native_world import ResidentWorld
from parliament_manager import ParliamentManager


@pytest.fixture
def native():
    native = configure_backend('cpp')
    try:
        yield native
    finally:
        configure_backend('python')


@functools.lru_cache(maxsize=1)
def reference():
    source = subprocess.check_output(['git', 'show', 'f2073f0:native_evaluation.py'], text=True)
    scope = {'__name__': 'frozen_a3_evaluation'}
    exec(compile(source, 'frozen-a3-native_evaluation.py', 'exec'), scope)
    return scope['evaluate_entities']


def population():
    model = TitanToPathModel()
    rows = [Pathstrider(str(i), np.arange(1., 13.) + i, titan_to_path_model=model) for i in range(9)]
    for i, entity in enumerate(rows):
        if i % 2 == 0:
            entity._path_affinities = np.roll(np.arange(1., 13.) / 78., i).copy()
        entity.titan_affinities = entity.titan_affinities[::-1].copy()
    rows[1].titan_affinities[:] = 0
    rows[3].is_titan_boss = True
    return rows


def parliament():
    manager = ParliamentManager()
    manager.seats = {name: i + 2 for i, name in enumerate(PATH_NAMES)}
    manager.suppression_factor = 0.7
    return manager


def rng_state():
    return pickle.dumps((random.getstate(), np.random.get_state())), torch.get_rng_state().numpy().tobytes()


def assert_same(actual, expected):
    for got, want in zip(actual, expected):
        for field in ('titan_affinities', '_path_affinities'):
            np.testing.assert_array_equal(getattr(got, field), getattr(want, field))
        for field in ('activity', 'stability', 'score', 'heroic_tendency'):
            assert type(getattr(got, field)) is type(getattr(want, field))
            assert getattr(got, field) == getattr(want, field)


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('distribution', [None, np.arange(1., 13.) / 78.])
def test_resident_scoring_uses_old_caches_and_keeps_exact_frozen_results(native, workers, distribution):
    actual, expected = population(), population()
    manager, old_manager = parliament(), parliament()
    originals = [(p.titan_affinities, p.titan_affinities.copy(), p._path_affinities,
                  None if p._path_affinities is None else p._path_affinities.copy()) for p in actual]
    evaluator, old_evaluator = native.BatchEvaluator(workers, 1), native.BatchEvaluator(workers, 1)
    world = ResidentWorld()
    before = rng_state()
    try:
        world.sync(actual)
        ids = world.stable_ids()
        reference()(expected, old_manager, distribution, evaluator=old_evaluator)
        importlib.import_module('native_evaluation').evaluate_entities(
            iter(actual), manager, distribution, evaluator=evaluator, world=world)
        assert world.scoring_count == 1, 'The normal call must actually execute resident scoring'
        assert world.stable_ids() == ids
        assert_same(actual, expected)
        assert manager._seat_cache_key == old_manager._seat_cache_key
        np.testing.assert_array_equal(manager._normalized_seat_vector, old_manager._normalized_seat_vector)
        for p, (titans, values, cache, cached_values) in zip(actual, originals):
            assert p.titan_affinities is titans
            np.testing.assert_array_equal(titans, values)
            if cache is not None:
                np.testing.assert_array_equal(cache, cached_values)
                assert p._path_affinities is not cache
            assert p._path_affinities.flags.owndata
        assert rng_state() == before
        with pytest.raises(RuntimeError, match='sync'):
            world.snapshot()
        world.sync(actual)
        np.testing.assert_array_equal(world.snapshot()['scores'], [p.score for p in actual])
    finally:
        evaluator.close()
        old_evaluator.close()
        world.close()


def test_resident_scoring_observes_direct_array_and_seat_cache_mutations(native):
    actual, expected = population(), population()
    manager, old_manager = parliament(), parliament()
    evaluator, old_evaluator = native.BatchEvaluator(), native.BatchEvaluator()
    world = ResidentWorld()
    try:
        for round_number in range(3):
            for rows, seats in ((actual, manager), (expected, old_manager)):
                rows[0].titan_affinities *= 1.25
                if round_number:
                    rows[2]._path_affinities[:] = np.arange(12.) / 66
                    seats._normalized_seat_vector[:] = np.linspace(0.01, 0.2, 12)
                if round_number == 2:
                    seats.seats[PATH_NAMES[1]] = 71
                    seats.total_seats = 17
                    seats.suppression_factor = -0.4
            reference()(expected, old_manager, None, evaluator=old_evaluator)
            importlib.import_module('native_evaluation').evaluate_entities(
                actual, manager, None, evaluator=evaluator, world=world)
            assert_same(actual, expected)
            assert manager._seat_cache_key == old_manager._seat_cache_key
        assert world.scoring_count == 3
        saved = [(p._path_affinities, p._path_affinities.copy()) for p in actual]
        world.close()
        for cached, values in saved:
            np.testing.assert_array_equal(cached, values)
    finally:
        evaluator.close()
        old_evaluator.close()
        world.close()


@pytest.mark.parametrize('case', ['raise_mode', 'duplicate', 'closed_evaluator', 'invalid_distribution'])
def test_unsupported_resident_scoring_preserves_frozen_error_prefix(native, case):
    actual, expected = population(), population()
    actual[0].titan_affinities[3] = expected[0].titan_affinities[3] = np.nan
    actual[1]._path_affinities = expected[1]._path_affinities = None
    if case == 'duplicate':
        actual = [actual[0], actual[0], actual[1]]
        expected = [expected[0], expected[0], expected[1]]
    manager, old_manager = parliament(), parliament()
    evaluator, old_evaluator = native.BatchEvaluator(), native.BatchEvaluator()
    world = ResidentWorld()
    distribution = np.ones(2) if case == 'invalid_distribution' else None
    if case == 'closed_evaluator':
        evaluator.close()
        old_evaluator.close()
    errors = []
    before = rng_state()
    try:
        with np.errstate(all='raise' if case == 'raise_mode' else 'ignore'):
            for rows, seats, engine, call, extra in (
                (expected, old_manager, old_evaluator, reference(), {}),
                (actual, manager, evaluator,
                 importlib.import_module('native_evaluation').evaluate_entities, {'world': world}),
            ):
                try:
                    call(rows, seats, distribution, evaluator=engine, **extra)
                    errors.append(None)
                except Exception as error:
                    errors.append(type(error))
        assert errors[0] is errors[1]
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got.titan_affinities, want.titan_affinities)
            if want._path_affinities is None:
                assert got._path_affinities is None
            else:
                np.testing.assert_array_equal(got._path_affinities, want._path_affinities)
            np.testing.assert_array_equal([got.score, got.activity], [want.score, want.activity])
        assert world.scoring_count == 0
        assert rng_state() == before
    finally:
        evaluator.close()
        old_evaluator.close()
        world.close()


def _assert_same_scoring_prefix(actual, expected):
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got.titan_affinities, want.titan_affinities)
        if want._path_affinities is None:
            assert got._path_affinities is None
        else:
            np.testing.assert_array_equal(got._path_affinities, want._path_affinities)
        for field in ('score', 'activity', 'stability', 'heroic_tendency'):
            assert type(getattr(got, field)) is type(getattr(want, field))
            assert getattr(got, field) == getattr(want, field)


@pytest.mark.parametrize('case', ['overflowed_cache', 'tiny_total'])
def test_default_numpy_warnings_promoted_to_errors_keep_scoring_prefix(native, case):
    import warnings

    actual, expected = population(), population()
    manager, old_manager = parliament(), parliament()
    # The original lazy getter publishes this cache before either failure.
    actual[0]._path_affinities = expected[0]._path_affinities = None
    if case == 'overflowed_cache':
        actual[1]._path_affinities = np.full(12, 1e308)
        expected[1]._path_affinities = np.full(12, 1e308)
    else:
        manager.total_seats = old_manager.total_seats = 1e-320
    evaluator, old_evaluator = native.BatchEvaluator(), native.BatchEvaluator()
    world = ResidentWorld()
    errors = []
    before = rng_state()
    try:
        # NumPy's default warn modes can still raise via Python's warning filter.
        with np.errstate(divide='warn', over='warn', under='ignore', invalid='warn'):
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                for rows, seats, engine, call, extra in (
                    (expected, old_manager, old_evaluator, reference(), {}),
                    (actual, manager, evaluator,
                     importlib.import_module('native_evaluation').evaluate_entities, {'world': world}),
                ):
                    try:
                        call(rows, seats, None, evaluator=engine, **extra)
                        errors.append(None)
                    except Exception as error:
                        errors.append((type(error), str(error)))
        assert errors[0] is not None and errors[0][0] is RuntimeWarning
        assert errors[1] == errors[0]
        assert expected[0]._path_affinities is not None
        _assert_same_scoring_prefix(actual, expected)
        assert manager._seat_cache_key == old_manager._seat_cache_key
        np.testing.assert_array_equal(manager._normalized_seat_vector, old_manager._normalized_seat_vector)
        assert world.scoring_count == 0
        assert rng_state() == before
    finally:
        evaluator.close()
        old_evaluator.close()
        world.close()


@pytest.mark.parametrize('override', ['instance', 'class'])
def test_custom_zeitgeist_method_retains_ordered_side_effects(native, monkeypatch, override):
    from types import MethodType

    # Capture production identities before installing intentional overrides.
    importlib.import_module('native_scoring')
    original = ParliamentManager.get_zeitgeist_multiplier
    actual, expected = population(), population()
    manager, old_manager = parliament(), parliament()
    rows_by_manager = {id(manager): actual, id(old_manager): expected}
    visits = {id(manager): [], id(old_manager): []}

    def custom(self, paths):
        visits[id(self)].append(paths.copy())
        self.suppression_factor -= 0.03
        # This changes a later entity's public cache during the original scan.
        rows_by_manager[id(self)][-1]._path_affinities *= 0.5
        return original(self, paths)

    if override == 'instance':
        manager.get_zeitgeist_multiplier = MethodType(custom, manager)
        old_manager.get_zeitgeist_multiplier = MethodType(custom, old_manager)
    else:
        monkeypatch.setattr(ParliamentManager, 'get_zeitgeist_multiplier', custom)
    evaluator, old_evaluator = native.BatchEvaluator(), native.BatchEvaluator()
    world = ResidentWorld()
    before = rng_state()
    try:
        reference()(expected, old_manager, None, evaluator=old_evaluator)
        importlib.import_module('native_evaluation').evaluate_entities(
            actual, manager, None, evaluator=evaluator, world=world)
        assert len(visits[id(manager)]) == len(actual)
        for got, want in zip(visits[id(manager)], visits[id(old_manager)]):
            np.testing.assert_array_equal(got, want)
        assert_same(actual, expected)
        assert manager.suppression_factor == old_manager.suppression_factor
        assert world.scoring_count == 0
        assert rng_state() == before
    finally:
        evaluator.close()
        old_evaluator.close()
        world.close()


def test_matching_public_seat_key_keeps_invalid_replacement_cache_and_failure_prefix(native):
    actual, expected = population(), population()
    manager, old_manager = parliament(), parliament()
    actual[0]._path_affinities = expected[0]._path_affinities = None
    for seats in (manager, old_manager):
        seats.get_zeitgeist_multiplier(np.ones(12) / 12)
        # Public replacement must not be silently repaired from the matching key.
        seats._normalized_seat_vector = np.ones(11)
    caches = manager._normalized_seat_vector, old_manager._normalized_seat_vector
    evaluator, old_evaluator = native.BatchEvaluator(), native.BatchEvaluator()
    world = ResidentWorld()
    errors = []
    before = rng_state()
    try:
        for rows, seats, engine, call, extra in (
            (expected, old_manager, old_evaluator, reference(), {}),
            (actual, manager, evaluator,
             importlib.import_module('native_evaluation').evaluate_entities, {'world': world}),
        ):
            try:
                call(rows, seats, None, evaluator=engine, **extra)
                errors.append(None)
            except Exception as error:
                errors.append((type(error), str(error)))
        assert errors[0] is not None and errors[0][0] is ValueError
        assert errors[1] == errors[0]
        assert expected[0]._path_affinities is not None
        _assert_same_scoring_prefix(actual, expected)
        assert manager._seat_cache_key == old_manager._seat_cache_key
        assert manager._normalized_seat_vector is caches[0]
        assert old_manager._normalized_seat_vector is caches[1]
        assert world.scoring_count == 0
        assert rng_state() == before
    finally:
        evaluator.close()
        old_evaluator.close()
        world.close()
