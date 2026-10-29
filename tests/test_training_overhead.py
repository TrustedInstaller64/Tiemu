"""Keep online learning identical while removing unused autograd work.

These tests execute real CPU networks and Adam updates. Only the external LLM
is replaced by a deterministic response; the reference interaction is extracted
unchanged from the fixed, pre-optimization commit on this same CI runner.
"""
import ast
import copy
from functools import lru_cache
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
import torch

import aeonic_cycle_manager as cycle_module
from aeonic_cycle_manager import AeonicCycleManager
from constants import TITAN_NAMES
from entities import Pathstrider
from models import ActionPolicyNetwork, ValueNetwork


ROOT = Path(__file__).resolve().parents[1]
BEFORE_OPTIMIZATION = "1ce39e00b8808564a331861ffcc799d4237ac324"


class FixedLlm:
    available = True

    def __init__(self, response):
        self.response = response

    def generate_response(self, prompt, max_tokens):
        return self.response


@lru_cache(maxsize=1)
def reference_interaction():
    source = subprocess.run(
        ["git", "show", f"{BEFORE_OPTIMIZATION}:aeonic_cycle_manager.py"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        check=True, timeout=30,
    ).stdout
    manager = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef)
                   and node.name == "AeonicCycleManager")
    method = next(node for node in manager.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "aeonic_interaction")
    namespace = dict(vars(cycle_module))
    # Execute only this unchanged method, not the old module or its imports.
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, f"{BEFORE_OPTIMIZATION}:aeonic_interaction", "exec"), namespace)
    return namespace["aeonic_interaction"]


def build_case(response):
    torch.manual_seed(193)
    action = ActionPolicyNetwork(24, 32)
    value = ValueNetwork(24, 32)
    manager = AeonicCycleManager(
        population_manager=None,
        action_policy_network=action,
        value_network=value,
        action_optimizer=torch.optim.Adam(action.parameters(), lr=0.0005),
        value_optimizer=torch.optim.Adam(value.parameters(), lr=0.001),
        value_loss_criterion=torch.nn.MSELoss(),
        titan_to_path_model_instance=None,
        existing_names=set(), name_to_entity_map={},
        llm_interface=FixedLlm(response) if response is not None else None,
        kaoselanna_llm_enabled=response is not None,
    )
    manager.aeonic_cycle_number = 3
    attacker = Pathstrider("attacker", np.linspace(0.25, 2.0, 12), "Reincarnator")
    attacker.score = 13.017
    defenders = []
    for index in range(4):
        defender = Pathstrider(f"defender-{index}", np.linspace(0.7, 1.5 + index, 12))
        defender.score = 7.031 + index * 1.0007
        if index % 2 == 0:
            defender.held_fire_seeds.add(TITAN_NAMES[index])
            defender.is_titan_form = TITAN_NAMES[index + 5]
        defenders.append(defender)
    return manager, attacker, defenders


def snapshot(manager, attacker, defenders, culled):
    return copy.deepcopy({
        "culled": None if culled is None else culled.name,
        "attacker_seeds": attacker.held_fire_seeds,
        "defenders": [(entity.name, entity.is_titan_form, entity.held_fire_seeds)
                      for entity in defenders],
        "defeated": manager.defeated_titans_this_cycle,
        "destruction_score": manager.baie_destruction_score,
        "seeds_collected": manager.baie_total_seeds_collected,
        "action": manager.action_policy_network.state_dict(),
        "value": manager.value_network.state_dict(),
        "action_optimizer": manager.action_optimizer.state_dict(),
        "value_optimizer": manager.value_optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    })


def trajectory(response, interaction):
    manager, attacker, defenders = build_case(response)
    initial_action = copy.deepcopy(manager.action_policy_network.state_dict())
    initial_value = copy.deepcopy(manager.value_network.state_dict())
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    states = []
    population = [attacker, *defenders]
    for defender in defenders:
        culled = interaction(manager, attacker, defender, attacker,
                             population, np.zeros(12))
        states.append(snapshot(manager, attacker, defenders, culled))
    assert any(not torch.equal(initial_action[name], parameter)
               for name, parameter in manager.action_policy_network.state_dict().items())
    assert any(not torch.equal(initial_value[name], parameter)
               for name, parameter in manager.value_network.state_dict().items())
    for optimizer in (manager.action_optimizer, manager.value_optimizer):
        assert {int(state["step"].item()) for state in optimizer.state.values()} == {4}
    return states


def assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for current, reference in zip(actual, expected):
            assert_exact(current, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("response", [
    None,
    json.dumps({"action": "谈判", "declaration": "test"}),
    json.dumps({"action": "击杀", "declaration": "test"}),
    "invalid JSON: use the original policy fallback",
])
def test_online_updates_keep_every_action_parameter_optimizer_and_rng_state(response):
    expected = trajectory(response, reference_interaction())
    actual = trajectory(response, AeonicCycleManager.aeonic_interaction)
    assert_exact(actual, expected)


@pytest.mark.parametrize("response", [None, "invalid JSON: policy fallback"])
def test_sampling_does_not_build_gradients_and_reuses_the_training_state(response):
    manager, attacker, defenders = build_case(response)
    action_calls, value_inputs = [], []

    def record_action(module, inputs, output):
        action_calls.append((torch.is_grad_enabled(), output.requires_grad, inputs[0]))

    def record_value(module, inputs, output):
        value_inputs.append(inputs[0])

    action_hook = manager.action_policy_network.register_forward_hook(record_action)
    value_hook = manager.value_network.register_forward_hook(record_value)
    try:
        manager.aeonic_interaction(attacker, defenders[0], attacker,
                                   [attacker, *defenders], np.zeros(12))
    finally:
        action_hook.remove()
        value_hook.remove()
    # Building a graph here wastes saved tensors: only the second forward trains.
    assert [(grad_enabled, requires_grad) for grad_enabled, requires_grad, _ in action_calls] == [
        (False, False), (True, True)]
    assert len(value_inputs) == 1
    assert action_calls[0][2] is action_calls[1][2] is value_inputs[0]
