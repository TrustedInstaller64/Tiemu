import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from reference_upstream import reference_result, replay_environment

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('backend,native_threads', [('python', 1), ('cpp', 1), ('cpp', 4)])
@pytest.mark.parametrize('reincarnator', [False, True])
def test_world_matches_frozen_upstream(backend, native_threads, reincarnator):
    cmd = [sys.executable, str(ROOT/'tests/replay_world.py'), '--root', str(ROOT), '--backend', backend,
           '--native-threads', str(native_threads), '--native-min-rows', '1']
    if reincarnator:
        cmd.append('--reincarnator')
    run = subprocess.run(cmd, cwd=ROOT, env=replay_environment(), capture_output=True,
                         text=True, encoding='utf-8', timeout=60)
    assert run.returncode == 0, run.stderr
    actual = json.loads(run.stdout)
    arguments = ('--reincarnator',) if reincarnator else ()
    expected = reference_result('replay_world.py', *arguments)
    for key in ('seats', 'random_state', 'numpy_state', 'torch_state'):
        assert actual[key] == expected[key], key
    assert len(actual['population']) == len(expected['population'])
    np.testing.assert_allclose(actual['blueprint'], expected['blueprint'], rtol=1e-6, atol=1e-7)
    for actual_entity, expected_entity in zip(actual['population'], expected['population']):
        for key, value in expected_entity.items():
            if key in ('titan_affinities', 'score', 'hp'):
                np.testing.assert_allclose(actual_entity[key], value, rtol=1e-10, atol=1e-10, err_msg=f'{actual_entity["name"]}/{key}')
            else:
                assert actual_entity[key] == value, (actual_entity['name'], key)
