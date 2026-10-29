"""Remote-only checks for shared batch path selection outside encounters."""
import copy

import numpy as np
import pytest

from entities import Pathstrider
import diversity_manager
import population_manager
import stagnation_manager
from test_election_batch_paths import native, population


def consumer(name):
    if name == 'distribution':
        manager = object.__new__(population_manager.PopulationManager)
        manager._tracked_population = None
        return population_manager, manager.get_global_path_distribution
    if name == 'stagnation':
        manager = stagnation_manager.StagnationManager(None, None, 0.01, 0.05)
        return stagnation_manager, manager.adjust_mutation_rate
    manager = diversity_manager.DiversityInterventionManager(
        {'diversity_intervention': {}}, None, None)
    return diversity_manager, manager._calculate_diversity_index


def assert_result_equal(name, actual, expected):
    if name == 'distribution':
        np.testing.assert_array_equal(actual, expected)
    elif name == 'diversity':
        assert actual[0] == expected[0]
        # Equal counts alone would miss insertion-order changes affecting ties.
        assert list(actual[1].items()) == list(expected[1].items())
        assert actual[1].most_common() == expected[1].most_common()
    else:
        assert actual == expected


@pytest.mark.parametrize('name', ['distribution', 'stagnation', 'diversity'])
@pytest.mark.parametrize('cached_ties', [False, True])
def test_shared_batch_consumers_match_original_scalar_order(native, monkeypatch, name, cached_ties):
    actual = population()
    if cached_ties:
        for entity, index in zip(actual, [3, 1, 3, 2, 1, 2]):
            entity._path_affinities = np.eye(12)[index].copy()
    expected = copy.deepcopy(actual)
    aliases = [p._path_affinities for p in actual]
    module, run = consumer(name)
    with monkeypatch.context() as patch:
        patch.setattr(module, '_native_dominant_paths', lambda *_: None)
        scalar_result = run(expected)
    batch_paths, entered = module._native_dominant_paths, []

    def record(*args):
        result = batch_paths(*args)
        entered.append(result is not None)
        return result

    monkeypatch.setattr(module, '_native_dominant_paths', record)
    assert_result_equal(name, run(actual), scalar_result)
    assert entered == [True]
    for index, (entity, reference) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(entity._path_affinities, reference._path_affinities)
        if aliases[index] is not None:
            assert entity._path_affinities is aliases[index]


@pytest.mark.parametrize('name', ['distribution', 'stagnation', 'diversity'])
def test_custom_getters_keep_original_order_and_calls(native, name):
    visits = []

    class CustomEntity(Pathstrider):
        @property
        def dominant_path_idx(self):
            visits.append(self.name)
            return np.int64(int(self.name) % 3)

    actual = [CustomEntity(str(i), np.ones(12)) for i in [2, 0, 1, 2]]
    _, run = consumer(name)
    run(actual)
    assert visits == ['2', '0', '1', '2']
    assert all(entity._path_affinities is None for entity in actual)


@pytest.mark.parametrize('raises', [False, True])
def test_temporary_distribution_histogram_is_closed_even_on_failure(native, monkeypatch, raises):
    closed, output = [], np.arange(12, dtype=np.float64) / 66

    class Histogram:
        def __init__(self, indices, paths):
            assert indices.shape == (6,)
            assert paths == 12

        def distribution(self):
            if raises:
                raise RuntimeError('distribution failed')
            return output

        def close(self):
            closed.append(True)

    monkeypatch.setattr(native, 'PathHistogram', Histogram)
    _, run = consumer('distribution')
    if raises:
        with pytest.raises(RuntimeError, match='distribution failed'):
            run(population())
    else:
        assert run(population()) is output
    assert closed == [True]


def test_tracked_distribution_does_not_use_shared_batch_helper(native, monkeypatch):
    actual = population()
    expected = np.zeros(12)
    for entity in copy.deepcopy(actual):
        expected[entity.dominant_path_idx] += 1
    expected /= len(actual)
    _, run = consumer('distribution')
    manager = run.__self__
    monkeypatch.setattr(population_manager, '_native_dominant_paths',
                        lambda *_: pytest.fail('tracked path must keep its existing lazy logic'))
    histogram = None
    try:
        with manager.track_distribution(actual):
            np.testing.assert_array_equal(run(actual), expected)
            histogram = manager._path_histogram
            np.testing.assert_array_equal(run(actual), expected)
        assert manager._tracked_population is None
        assert manager._path_histogram is None
    finally:
        if histogram is not None:
            histogram.close()
