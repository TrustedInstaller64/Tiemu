"""Compare real phase transitions and trained networks with frozen upstream."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def replay(backend, phase, population, tmp_path):
    process = subprocess.run(
        [sys.executable, "-B", str(ROOT / "tests" / "replay_phases.py"),
         "--root", str(ROOT), "--backend", backend, "--phase", phase,
         "--population", str(population)],
        cwd=tmp_path, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
        capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert not list(tmp_path.rglob("*.pth")), "The replay must not export models"
    return json.loads(process.stdout)


def assert_state_equal(actual, expected, tolerance=1e-10, location="state"):
    """Keep discrete state exact while allowing bounded native arithmetic error."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), location
        for key, value in expected.items():
            assert_state_equal(actual[key], value, tolerance, f"{location}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), location
        if expected and all(isinstance(value, float) for value in expected):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, err_msg=location)
        else:
            for index, (current, reference) in enumerate(zip(actual, expected)):
                assert_state_equal(current, reference, tolerance, f"{location}[{index}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=tolerance, abs=tolerance), location
    else:
        assert actual == expected, location


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_inorganic_seventeen_generations_match_upstream_legacy_and_rng(backend, tmp_path):
    expected = json.loads((FIXTURES / "upstream-inorganic.json").read_text(encoding='utf-8'))
    actual = replay(backend, "inorganic", 100, tmp_path)
    assert actual["generations"] == 17
    assert actual["rng"] == expected["rng"]
    assert_state_equal(actual, expected)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("population", [100, 200])
def test_aeonic_two_generations_match_upstream_world_rng_and_real_training(backend, population, tmp_path):
    expected = json.loads((FIXTURES / f"upstream-aeonic{population}.json").read_text(encoding='utf-8'))
    actual = replay(backend, "aeonic", population, tmp_path)
    actual_networks, expected_networks = actual.pop("networks"), expected.pop("networks")
    assert len(actual["outcomes"]) == 2
    assert actual["rng"] == expected["rng"]
    assert_state_equal(actual, expected)
    assert_state_equal(actual_networks, expected_networks, tolerance=1e-6, location="networks")
    for name, network in actual_networks.items():
        assert network["changed_tensors"] > 0, name
        assert network["maximum_parameter_change"] > 0, name
        assert min(network["optimizer_steps"]) > 0, name
        assert not np.array_equal(network["before"], network["after"]), name
