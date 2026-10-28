import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('backend', ['python', 'cpp'])
@pytest.mark.parametrize('reincarnator', [False, True])
def test_world_matches_frozen_upstream(backend, reincarnator):
    cmd = [sys.executable, str(ROOT/'tests/replay_world.py'), '--root', str(ROOT), '--backend', backend]
    if reincarnator:
        cmd.append('--reincarnator')
    run = subprocess.run(cmd, cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    actual = json.loads(run.stdout)
    fixture = 'upstream-guide.json' if reincarnator else 'upstream-human.json'
    expected = json.loads((ROOT/'tests/fixtures'/fixture).read_text(encoding='utf-8'))
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
