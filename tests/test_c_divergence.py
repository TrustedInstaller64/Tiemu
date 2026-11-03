"""Diagnostic contracts, exercised on hosted runners only."""
import copy
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'benchmarks' / 'c_divergence.py'
pytestmark = pytest.mark.xfail(not MODULE_PATH.exists(), strict=True,
                              raises=ModuleNotFoundError,
                              reason='C1 diagnostic comparison module not implemented yet')

@pytest.fixture
def compare():
    path = MODULE_PATH
    if not path.exists():
        raise ModuleNotFoundError('C1 diagnostic comparison module not implemented yet')
    spec = importlib.util.spec_from_file_location('c_divergence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_runs


def state(names=('a', 'b'), scores=(1., 3.)):
    return {'population': [dict(name=n, score=s, hp=100., titan_affinities=[1., 0.],
                               cached_paths=[1., 0.], trait='Mortal', golden_one_tenure=0,
                               is_titan_boss=False, titan_aspect=None, held_fire_seeds=[])
                           for n, s in zip(names, scores)],
            'dominant_path_counts': [len(names), 0], 'seats': {'x': 2, 'y': 0},
            'culled_names': [], 'blueprint': [1., 0.], 'zeitgeist': [0.5, 0.5],
            'rng': {'python': 'p', 'numpy': 'n', 'torch': 't'},
            'networks': {'guide': 'hash'}, 'managers': {}}


def run(states):
    return {'meta': {'seed': 731, 'population': 2, 'generations': len(states)-1,
                     'reincarnator': False, 'observed': True},
            'snapshots': [{'step': i, 'stage': 'initial' if not i else 'after_growth', 'state': s}
                          for i, s in enumerate(states)],
            'final_state': states[-1], 'final_network_values': {}}


def test_identical_reports_zero_without_claiming_unbounded_equivalence(compare):
    report = compare(run([state(), state()]), run([state(), state()]))
    assert report['first_numeric_difference'] is None
    assert report['first_discrete_difference'] is None
    assert report['final']['numeric']['score']['max_absolute_error'] == 0
    assert report['final']['path_total_variation'] == 0
    assert report['final']['score_distribution']['ks_distance'] == 0


def test_numeric_and_discrete_first_difference_are_separate(compare):
    a = run([state(), state(), state()])
    b = copy.deepcopy(a)
    b['snapshots'][1]['state']['population'][0]['score'] += 1e-9
    b['snapshots'][2]['state']['seats'] = {'x': 1, 'y': 1}
    result = compare(a, b)
    assert result['first_numeric_difference']['step'] == 1
    assert result['first_discrete_difference']['step'] == 2
    assert result['first_differences']['seats']['step'] == 2
    assert result['final']['seat_l1'] == 2


def test_order_change_not_hidden_by_name_sorting(compare):
    a, b = run([state()]), run([state()])
    b['snapshots'][0]['state']['population'].reverse()
    result = compare(a, b)
    assert result['first_differences']['population_order']['step'] == 0
    assert result['final']['common_name_count'] == 2


def test_membership_loss_has_coverage_and_no_false_postfork_identity(compare):
    a, b = run([state(), state()]), run([state(), state(('a', 'c'), (1., 8.))])
    result = compare(a, b)
    assert result['final']['common_name_count'] == 1
    assert result['final']['name_jaccard'] == pytest.approx(1/3)
    assert result['final']['individual_comparison_valid'] is False
    assert result['final']['numeric'] is None
    assert result['final']['score_distribution']['ks_distance'] == 0.5


def test_histogram_tv_and_score_wasserstein_are_normalized(compare):
    a, b = run([state()]), run([state(scores=(3., 5.))])
    b['final_state']['dominant_path_counts'] = [0, 2]
    result = compare(a, b)
    assert result['final']['path_total_variation'] == 1
    assert result['final']['score_distribution']['wasserstein_1'] == 2
    assert result['final']['score_distribution']['reference']['mean'] == 2
    assert result['final']['score_distribution']['candidate']['mean'] == 4


def test_nonfinite_values_are_counted_not_silently_dropped(compare):
    a, b = run([state()]), run([state(scores=(float('nan'), float('inf')))])
    result = compare(a, b)
    distribution = result['final']['score_distribution']
    assert distribution['candidate']['nonfinite_count'] == 2
    assert distribution['candidate']['mean'] is None
    assert distribution['ks_distance'] is None
    assert result['final']['numeric']['score']['nonfinite_pair_count'] == 2


def test_rng_and_network_changes_have_independent_first_locations(compare):
    a, b = run([state(), state()]), run([state(), state()])
    b['snapshots'][1]['state']['rng']['numpy'] = 'changed'
    b['snapshots'][1]['state']['networks']['guide'] = 'changed'
    result = compare(a, b)
    assert result['first_differences']['rng']['step'] == 1
    assert result['first_differences']['networks']['step'] == 1


def test_incompatible_snapshots_fail_instead_of_truncating_zip(compare):
    with pytest.raises(ValueError, match='snapshot'):
        compare(run([state(), state()]), run([state()]))


def test_duplicate_names_are_rejected(compare):
    with pytest.raises(ValueError, match='duplicate'):
        compare(run([state()]), run([state(('a', 'a'))]))


def test_scalar_zero_reference_has_defined_scaled_error(compare):
    a, b = run([state(scores=(0., 0.))]), run([state(scores=(1., 0.))])
    result = compare(a, b)
    assert result['final']['numeric']['score']['max_absolute_error'] == 1
    assert result['final']['numeric']['score']['relative_l2_error'] is None


def test_first_numeric_report_is_censored_after_discrete_fork(compare):
    a, b = run([state()]), run([state(('a', 'c'))])
    result = compare(a, b)
    assert result['first_numeric_difference'] is None
    assert result['individual_first_numeric_location_censored'] is True
    assert result['individual_numeric_comparison_stopped_at']['step'] == 0


def test_reporting_dominant_path_is_separate_from_real_decisions(compare):
    a, b = run([state()]), run([state()])
    a['final_state']['population'][0]['reporting_dominant_idx'] = 0
    b['final_state']['population'][0]['reporting_dominant_idx'] = 1
    result = compare(a, b)
    assert result['first_differences']['reporting_path']['step'] == 0
    assert result['first_discrete_difference'] is None
