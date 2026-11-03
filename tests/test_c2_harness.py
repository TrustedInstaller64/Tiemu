"""Cloud-only measurement contracts: scope, flags, attribution and birth identity."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def harness(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/'benchmarks'
    monkeypatch.syspath_prepend(str(directory))
    def load(name):
        spec = importlib.util.spec_from_file_location(name, directory/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load


def test_original_suite_is_bounded_and_large_is_explicit(harness):
    module = harness('compare_c2')
    assert module.selected_cases('paired') == [(1000, 6, False), (1000, 6, True)]
    assert module.selected_cases('large-native') == [(10000, 6, False)]
    with pytest.raises(ValueError):
        module.selected_cases('unbounded')


def test_c1_command_remains_scoring_only_and_c2_is_orthogonal(harness, tmp_path):
    module = harness('compare_c2')
    configs = {row['name']: row for row in module.configurations(tmp_path, tmp_path, tmp_path, tmp_path)}
    for name in ('original', 'a2', 'c1', 'off', 'c1only'):
        command = module.command_for('runner.py', configs[name], 'out.json', 1000, 6, 731, False, False)
        assert '--experimental-c-newborns' not in command
    for name in ('c2_1', 'c2_2', 'c2_4'):
        command = module.command_for('runner.py', configs[name], 'out.json', 1000, 6, 731, True, True)
        assert '--experimental-c-newborns' in command
        assert command[command.index('--mode')+1] == 'c'
        assert command[command.index('--native-min-rows')+1] == '1'
        assert '--reincarnator' in command and '--observe' in command


def test_disabled_features_use_two_distinct_frozen_oracles(harness):
    module = harness('compare_c2')
    values = {'a2': 'A', 'off': 'A', 'c1': 'C', 'c1only': 'C',
              'c2_1': 'new', 'c2_2': 'new', 'c2_4': 'fork'}
    gates = module.identity_gates({name: {'final_state_digest': value} for name, value in values.items()})
    assert gates == {'off_matches_a2': True, 'c1only_matches_frozen_c1': True,
                     'c2_thread_digests_equal': False}


def test_native_only_report_cannot_manufacture_original_speed(harness):
    module = harness('compare_c2')
    samples = {name: [{'wall_seconds': value}] for name, value in
               {'a2': 12., 'c1': 10., 'off': 12., 'c1only': 9.,
                'c2_1': 6., 'c2_2': 4., 'c2_4': 3.}.items()}
    result = module.speed_report(samples)
    assert result['speedup_over_original'] is None
    assert result['speedup_over_c1']['c2_1'] == pytest.approx(10/6)
    assert result['speedup_over_c1only']['c2_1'] == 1.5
    assert result['speedup_over_c2_one_thread']['c2_4'] == 2


def test_birth_contract_checks_type_cache_and_array_ownership(harness):
    module = harness('benchmark_c2_births')
    entity = SimpleNamespace(name='new', score=0, _path_affinities=None, titan_affinities=np.ones(12))
    manager = SimpleNamespace(name_to_entity_map={'new': entity}, existing_names={'new'})
    good = module.birth_contract([entity], [entity], manager)
    assert all(value for key, value in good.items() if key not in ('count', 'names'))
    entity.score = 0.0
    entity._path_affinities = np.ones(12)
    entity.titan_affinities = np.ones((2, 12))[0]
    bad = module.birth_contract([entity], [entity], manager)
    assert bad['integer_zero_scores'] is False
    assert bad['empty_path_caches'] is False
    assert bad['owned_arrays'] is False


def test_birth_contract_detects_shared_storage_and_wrong_registry(harness):
    module = harness('benchmark_c2_births')
    values = np.ones(12)
    a = SimpleNamespace(name='a', score=0, _path_affinities=None, titan_affinities=values)
    b = SimpleNamespace(name='b', score=0, _path_affinities=None, titan_affinities=values)
    manager = SimpleNamespace(name_to_entity_map={'a': b, 'b': b}, existing_names={'a', 'b', 'stale'})
    result = module.birth_contract([a, b], [a, b], manager)
    assert result['disjoint_arrays'] is False
    assert result['map_matches'] is False
    assert result['registry_matches_population'] is False
