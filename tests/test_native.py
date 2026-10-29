"""Differential contracts for the native core, independent of application boot."""
import importlib
import numpy as np
import pytest


@pytest.fixture(scope="module")
def native():
    try:
        return importlib.import_module("_tiemu_ffi")
    except ImportError as exc:
        pytest.fail(f"Build the native extension before running native tests: {exc}")


def reference_paths(values, matrix):
    def row(v):
        result = np.maximum(0, np.dot(v, matrix))
        total = np.sum(result)
        return result / total if total > 0 else result
    return row(values) if values.ndim == 1 else np.array([row(v) for v in values])


def reference_inorganic(activity, stability, noise, grid):
    # The original shallow-copied list shares dictionaries: later cells see
    # earlier updates in the same generation. A simultaneous update is wrong.
    cells = [dict(activity=a, stability=s) for a, s in zip(activity, stability)]
    next_cells = list(cells)
    for i in range(grid):
        for j in range(grid):
            idx = i * grid + j
            neighbors = [cells[(i + di) * grid + j + dj]
                         for di in (-1, 0, 1) for dj in (-1, 0, 1)
                         if (di or dj) and 0 <= i + di < grid and 0 <= j + dj < grid]
            for channel, field in enumerate(("activity", "stability")):
                average = sum(c[field] for c in neighbors) / len(neighbors) if neighbors else 0
                change = (average - cells[idx][field]) * 0.1 + noise[idx, channel]
                next_cells[idx][field] = max(0, cells[idx][field] + change)
    return tuple(np.array([c[f] for c in next_cells]) for f in ("activity", "stability"))


@pytest.mark.parametrize("batch", [False, True])
def test_path_mapping_normalizes_each_entity_not_whole_batch(native, batch):
    rng = np.random.default_rng(731)
    values = rng.uniform(0, 5, (17, 12))
    matrix = rng.uniform(-1, 1, (12, 12))
    if not batch:
        values = values[0]
    before = values.copy()
    actual = native.path_affinities(values, matrix)
    np.testing.assert_allclose(actual, reference_paths(values, matrix), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(actual.sum(axis=-1), 1)
    np.testing.assert_array_equal(values, before)


def test_path_mapping_zero_and_negative_paths_stay_zero(native):
    np.testing.assert_array_equal(native.path_affinities(np.ones(3), -np.ones((3, 2))), [0, 0])
    assert native.path_affinities(np.empty((0, 3)), np.ones((3, 2))).shape == (0, 2)


def test_path_mapping_accepts_strided_views_without_mutating_them(native):
    values = np.arange(48, dtype=float).reshape(4, 12)[:, ::2]
    matrix = np.eye(12)[::2, ::2]
    np.testing.assert_allclose(native.path_affinities(values, matrix), reference_paths(values, matrix))


@pytest.mark.parametrize("distribution,boss", [(None, False), (np.array([0.2] * 12), False), (np.array([0.2] * 12), True)])
def test_concepts_preserves_l1_purity_saturation_and_boss_multiplier(native, distribution, boss):
    paths = np.array([0.3, 0.4, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0])
    expected_score = 8 * 1.4 * 1.1
    if distribution is not None:
        expected_score /= 1.4
    if boss:
        expected_score *= 5
    np.testing.assert_allclose(native.concepts(paths, 1.1, distribution, boss), [5, 3, 0.4, expected_score], rtol=2e-14)


def test_concepts_zero_and_tie_use_first_maximum(native):
    np.testing.assert_array_equal(native.concepts(np.zeros(12), 1, None, False), [0, 0, 0, 0])
    values = np.array([0.5, 0.5] + [0.] * 10)
    distribution = np.array([0.1, 0.8] + [0.] * 10)
    expected = np.linalg.norm(values[:6]) * 10 * 1.5 / 1.2
    assert native.concepts(values, 1, distribution, False)[3] == pytest.approx(expected)


@pytest.mark.parametrize("values,factor,expected", [
    ([2., 1., 0.], .01, [2., .98, 0.]),
    ([2., 2., 1.], .1, [2., 1.8, .8]),
    ([2., 1., 0.], 0, [2., 1., 0.]),
    ([2., 1., -1.], -1, [2., 1., -1.]),
    ([2.], 1, [2.]),
    ([], 1, []),
])
def test_purify_subtracts_dominant_value_and_keeps_first_tie(native, values, factor, expected):
    original = np.array(values)
    actual = native.purify(original, factor)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(original, values)


@pytest.mark.parametrize("values,limit", [([3., 4.], 10), ([3., 4.], 2), ([0., 0.], 10), ([np.inf, 1.], 10), ([np.nan, 1.], 10), ([1e200, 1e200], 10)])
def test_normalize_is_a_cap_and_retains_original_nonfinite_branch(native, values, limit):
    values = np.array(values)
    with np.errstate(all="ignore"):
        norm = np.linalg.norm(values)
        expected = values / (norm + 1e-9) * limit if not np.isfinite(norm) or norm == 0 or norm > limit else values
    np.testing.assert_allclose(native.normalize(values, limit), expected, equal_nan=True)


def test_synergy_uses_updated_first_entity_for_second_update(native):
    a, b = np.array([2., 1.]), np.array([1., 2.])
    actual_a, actual_b = native.synergy(a, b)
    np.testing.assert_allclose(actual_a, [2.02, 1.04])
    np.testing.assert_allclose(actual_b, [1.0404, 2.0208])
    np.testing.assert_array_equal(a, [2., 1.])
    np.testing.assert_array_equal(b, [1., 2.])


def test_clash_uses_original_loser_influence_for_both_entities(native):
    winner, loser = np.array([2., 1.]), np.array([1., 2.])
    actual_winner, actual_loser = native.clash(winner, loser, .7)
    np.testing.assert_allclose(actual_winner, [2.07, 1.14])
    np.testing.assert_allclose(actual_loser, [.965, 1.93])
    np.testing.assert_array_equal(winner, [2., 1.])
    np.testing.assert_array_equal(loser, [1., 2.])


def test_mentorship_uses_updated_weaker_for_feedback(native):
    stronger, weaker = np.array([2., 1.]), np.array([1., 2.])
    # atanh(.5) makes influence exactly .025, with hand-derived sequential result
    actual_stronger, actual_weaker = native.mentorship(stronger, weaker, 1 + np.arctanh(.5))
    np.testing.assert_allclose(actual_weaker, [1.025, 1.975])
    np.testing.assert_allclose(actual_stronger, [1.995125, 1.004875])
    np.testing.assert_array_equal(stronger, [2., 1.])
    np.testing.assert_array_equal(weaker, [1., 2.])


def test_repulsion_uses_first_dominant_and_provided_noise_after_damage(native):
    values = np.array([2., 2., 1.])
    noise = np.array([.5, .25, 1.])
    np.testing.assert_allclose(native.repulsion(values, noise), [1.905, 2.0025, 1.01])
    np.testing.assert_array_equal(values, [2., 2., 1.])
    np.testing.assert_array_equal(noise, [.5, .25, 1.])


def test_interaction_vectors_match_original_numpy_expressions(native):
    rng = np.random.default_rng(971)
    for _ in range(20):
        a, b = rng.uniform(0, 1000, (2, 12))
        left, right = a.copy(), b.copy()
        left += right * .02
        right += left * .02
        np.testing.assert_allclose(native.synergy(a, b), (left, right), rtol=2e-14, atol=2e-14)
        influence = b * .1 * .7
        np.testing.assert_allclose(native.clash(a, b, .7), (a + influence, b - influence * .5), rtol=2e-14, atol=2e-14)
        weaker = b + (a - b) * (.05 * np.tanh(1.5 - 1))
        stronger = a + (weaker - a) * .005
        np.testing.assert_allclose(native.mentorship(a, b, 1.5), (stronger, weaker), rtol=2e-14, atol=2e-14)
        expected = a.copy()
        damage = expected[np.argmax(expected)] * .05
        expected[np.argmax(expected)] -= damage
        noise = rng.random(12)
        expected += noise * damage * .1
        np.testing.assert_allclose(native.repulsion(a, noise), expected, rtol=2e-14, atol=2e-14)


def test_vote_totals_preserves_signed_weights_identity_preweighting_and_order(native):
    logits = np.array([[1., -1., .2], [.5, 1., -.7], [.2, .2, 1.]])
    dom = np.array([0, 1, 0, 2, 1], dtype=np.int64)
    weights = np.array([8., 5., 1., -0.5, 0.])
    expected = np.zeros(3)
    for d, weight in zip(dom, weights):
        row = np.exp(logits[d] - np.max(logits[d]))
        expected += row / row.sum() * weight
    for _ in range(3):
        np.testing.assert_allclose(native.vote_totals(dom, weights, logits), expected, rtol=2e-14)
    np.testing.assert_array_equal(native.vote_totals(np.array([], dtype=np.int64), np.array([]), logits), [0, 0, 0])


@pytest.mark.parametrize("grid", [1, 2, 10])
def test_inorganic_step_keeps_original_sequential_updates_and_noise_order(native, grid):
    rng = np.random.default_rng(917)
    act = rng.uniform(1, 5, grid * grid)
    stab = rng.uniform(1, 5, grid * grid)
    before_act, before_stab = act.copy(), stab.copy()
    for _ in range(7):
        noise = rng.uniform(-.1, .1, (grid * grid, 2))
        expected = reference_inorganic(act, stab, noise, grid)
        actual = native.inorganic_step(act, stab, noise, grid)
        np.testing.assert_array_equal(act, before_act)
        np.testing.assert_array_equal(stab, before_stab)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        act, stab = actual
        before_act, before_stab = act.copy(), stab.copy()


def test_histogram_updates_counts_without_changing_population_order(native):
    histogram = native.PathHistogram(np.array([0, 2, 2, 1], dtype=np.int64), 3)
    np.testing.assert_array_equal(histogram.distribution(), [.25, .25, .5])
    histogram.update(2, 0)
    histogram.update(2, 0)
    np.testing.assert_array_equal(histogram.distribution(), [.5, .25, .25])
    snapshot = histogram.distribution()
    snapshot[:] = 0
    np.testing.assert_array_equal(histogram.distribution(), [.5, .25, .25])
    np.testing.assert_allclose(native.PathHistogram(np.array([], dtype=np.int64), 3).distribution(), [1 / 3] * 3)


def test_indices_do_not_silently_truncate_floating_point_path_ids(native):
    with pytest.raises(TypeError):
        native.PathHistogram(np.array([1.9]), 3)
    with pytest.raises(TypeError):
        native.vote_totals(np.array([1.9]), np.ones(1), np.eye(3))


def test_opponent_index_preserves_bucket_order_stale_dead_candidates_and_fallback(native):
    index = native.OpponentIndex(np.array([20., 0., 10., 21., 100.]), 10.)
    assert index.alive_count == 5
    assert [index.entity_at(i) for i in range(5)] == [0, 1, 2, 3, 4]
    # Bucket 0, then bucket 1, then bucket 2; exclude only self.
    assert [index.opponent_at(2, 10., i) for i in range(index.opponent_count(2, 10.))] == [1, 0, 3]
    index.remove(0)
    index.remove(0)
    assert index.alive_count == 4
    assert [index.entity_at(i) for i in range(4)] == [1, 2, 3, 4]
    assert [index.opponent_at(2, 10., i) for i in range(index.opponent_count(2, 10.))] == [1, 0, 3]
    assert [index.opponent_at(4, 100., i) for i in range(index.opponent_count(4, 100.))] == [1, 2, 3]
    # Buckets are fixed at construction; query uses the current score.
    assert [index.opponent_at(2, 100., i) for i in range(index.opponent_count(2, 100.))] == [4]


def test_opponent_index_matches_python_float_floor_division_and_empty_pool(native):
    index = native.OpponentIndex(np.array([1., .9, -0.1]), .1)
    bins = {}
    for i, score in enumerate([1., .9, -0.1]):
        bins.setdefault(int(score // .1), []).append(i)
    for entity, score in enumerate([1., .9, -0.1]):
        bucket = int(score // .1)
        pool = [i for b in range(bucket - 1, bucket + 2) for i in bins.get(b, []) if i != entity]
        pool = pool or [i for i in range(3) if i != entity]
        assert [index.opponent_at(entity, score, r) for r in range(index.opponent_count(entity, score))] == pool
    singleton = native.OpponentIndex(np.array([2.]), 10.)
    assert singleton.opponent_count(0, 2.) == 0
    singleton.remove(0)
    assert singleton.alive_count == 0


def test_opponent_ranks_match_reference_after_many_removals_and_score_changes(native):
    rng = np.random.default_rng(131)
    scores = rng.uniform(-30, 130, 80)
    index = native.OpponentIndex(scores, 10.)
    buckets = {}
    for i, score in enumerate(scores):
        buckets.setdefault(int(score // 10), []).append(i)
    alive = list(range(len(scores)))
    for step in range(90):
        if step < 79:
            removed = int(rng.choice(alive))
            index.remove(removed)
            alive.remove(removed)
        assert index.alive_count == len(alive)
        assert [index.entity_at(rank) for rank in range(len(alive))] == alive
        entity = int(rng.choice(alive))
        current_score = float(rng.uniform(-100, 200))
        current_bin = int(current_score // 10)
        candidates = [i for b in range(current_bin - 1, current_bin + 2)
                      for i in buckets.get(b, []) if i != entity]
        if not candidates:
            candidates = [i for i in alive if i != entity]
        assert index.opponent_count(entity, current_score) == len(candidates)
        assert [index.opponent_at(entity, current_score, rank)
                for rank in range(len(candidates))] == candidates


@pytest.mark.parametrize("operation", [
    lambda n: n.path_affinities(np.ones((2, 3, 4)), np.eye(4)),
    lambda n: n.path_affinities(np.ones(3), np.eye(4)),
    lambda n: n.path_affinities(np.ones(3), np.ones(3)),
    lambda n: n.concepts(np.ones((1, 12)), 1., None, False),
    lambda n: n.concepts(np.ones(12), 1., np.ones(11), False),
    lambda n: n.concepts(np.array([]), 1., None, False),
    lambda n: n.purify(np.ones((2, 3)), .1),
    lambda n: n.normalize(np.ones((2, 3)), 10.),
    lambda n: n.synergy(np.ones(2), np.ones(3)),
    lambda n: n.clash(np.ones((1, 2)), np.ones(2), .7),
    lambda n: n.mentorship(np.ones(2), np.ones(3), 1.5),
    lambda n: n.repulsion(np.array([]), np.array([])),
    lambda n: n.repulsion(np.ones(2), np.ones(3)),
    lambda n: n.vote_totals(np.array([3]), np.ones(1), np.eye(3)),
    lambda n: n.vote_totals(np.array([-1]), np.ones(1), np.eye(3)),
    lambda n: n.vote_totals(np.array([0, 1]), np.ones(1), np.eye(3)),
    lambda n: n.inorganic_step(np.ones(4), np.ones(4), np.ones((4, 2)), 3),
    lambda n: n.inorganic_step(np.ones(4), np.ones(4), np.ones((2, 4)), 2),
    lambda n: n.inorganic_step(np.ones(1), np.ones(1), np.ones((1, 2)), 0),
    lambda n: n.PathHistogram(np.array([0]), 0),
    lambda n: n.PathHistogram(np.array([3]), 3),
    lambda n: n.PathHistogram(np.array([0]), 3).update(1, 0),
    lambda n: n.PathHistogram(np.array([0]), 3).update(0, -1),
    lambda n: n.OpponentIndex(np.array([1.]), 0.),
    lambda n: n.OpponentIndex(np.array([np.nan]), 10.),
    lambda n: n.OpponentIndex(np.array([1.]), 10.).entity_at(1),
    lambda n: n.OpponentIndex(np.array([1.]), 10.).opponent_at(0, 1., 0),
    lambda n: n.OpponentIndex(np.array([1.]), 10.).opponent_count(1, 1.),
    lambda n: n.OpponentIndex(np.array([1.]), 10.).remove(-1),
])
def test_malformed_shapes_and_indices_fail_before_native_memory_access(native, operation):
    with pytest.raises((ValueError, IndexError)):
        operation(native)


def test_backend_selection_is_explicit_and_missing_cpp_does_not_silently_fallback(monkeypatch):
    try:
        backend = importlib.import_module("native_backend")
    except ImportError as exc:
        pytest.fail(f"native_backend is not implemented: {exc}")
    real_import = importlib.import_module
    def missing_native(name, *args, **kwargs):
        if name == "_tiemu_ffi":
            raise ImportError("deliberately absent native extension")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", missing_native)
    backend.configure_backend("python")
    assert backend.get_native() is None
    with pytest.warns(RuntimeWarning, match="native|C\\+\\+"):
        backend.configure_backend("auto")
    assert backend.get_native() is None
    with pytest.raises(RuntimeError, match="native|C\\+\\+"):
        backend.configure_backend("cpp")
    with pytest.raises(ValueError):
        backend.configure_backend("typo")
    backend.configure_backend("python")


def test_cpp_backend_returns_actual_extension(native):
    backend = importlib.import_module("native_backend")
    try:
        backend.configure_backend("cpp")
        assert backend.get_native() is native
    finally:
        backend.configure_backend("python")
