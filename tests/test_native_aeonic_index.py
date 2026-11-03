"""Cloud-only contracts for ordered aeonic candidates and callback refreshes."""
import ast
import copy
import functools
import importlib
import pickle
import random
import subprocess
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

import aeonic_cycle_manager as cycle_module
from aeonic_cycle_manager import AeonicCycleManager
from constants import TITAN_NAMES
from entities import Pathstrider
from models import ActionPolicyNetwork, TitanToPathModel, ValueNetwork
from native_backend import configure_backend
from parliament_manager import ParliamentManager
from population_manager import PopulationManager


ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'f2073f0'


def candidate():
    try:
        return importlib.import_module('native_aeonic')
    except ImportError as error:
        pytest.fail(f'Native aeonic candidate bridge has not been implemented: {error}')


@pytest.fixture
def native():
    configure_backend('cpp')
    try:
        yield
    finally:
        configure_backend('python')


@functools.lru_cache(maxsize=1)
def frozen():
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                      f'{BASELINE}:aeonic_cycle_manager.py'], text=True)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'AeonicCycleManager')
    names = ('select_aeonic_opponents', 'aeonic_interaction', 'run_aeonic_cycle_generation')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(vars(cycle_module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 'frozen-A3-aeonic-methods', 'exec'), scope)
    return {name: scope[name] for name in names}


def make_case(kind='mixed', size=24):
    torch.manual_seed(211)
    model = TitanToPathModel()
    population = [Pathstrider(f'entity-{i}', np.arange(1., 13.) + i / 10,
                              titan_to_path_model=model) for i in range(size)]
    names = {p.name for p in population}
    mapping = {p.name: p for p in population}
    manager = PopulationManager(names, mapping, np.ones(12), .05, .01, 12, .1,
                                 model, 100., 1., 1000.)
    actor, value = ActionPolicyNetwork(24, 16), ValueNetwork(24, 16)
    cycle = AeonicCycleManager(manager, actor, value,
                                torch.optim.Adam(actor.parameters(), lr=.0005),
                                torch.optim.Adam(value.parameters(), lr=.001), torch.nn.MSELoss(),
                                model, names, mapping, parliament_manager=ParliamentManager())
    cycle.aeonic_cycle_number = 2
    for i, p in enumerate(population):
        p.score = float(2 + i % 7)
        if i % 3 == 0:
            p.trait = 'GoldenOne'
        if i % 4 == 1:
            p.is_titan_form = TITAN_NAMES[i % 12]
        if i % 5 == 2:
            p.held_fire_seeds.add(TITAN_NAMES[i % 12])
    reincarnator = population[0] if population else None
    if kind == 'reincarnator-golden':
        reincarnator.trait = 'GoldenOne'
    elif kind == 'outside':
        reincarnator = Pathstrider('outside', np.ones(12), titan_to_path_model=model)
    elif kind in ('plain', 'titans', 'ordinary'):
        reincarnator = None
        for i, p in enumerate(population):
            p.trait = 'GoldenOne' if kind == 'ordinary' else 'Mortal'
            p.is_titan_form = TITAN_NAMES[i % 12] if kind == 'titans' and i % 2 else None
            p.held_fire_seeds = {TITAN_NAMES[i % 12]} if kind == 'ordinary' else set()
    elif kind == 'reincarnator-only':
        for p in population:
            p.trait = 'Mortal'
        reincarnator.trait = 'Reincarnator'
    if population and kind != 'ordinary':
        # Empty strings are false in the original candidate predicates, unlike
        # the separate `is not None` test used inside ordinary interactions.
        population[-1].is_titan_form = ''
    return cycle, population, reincarnator


def rng_state():
    return pickle.dumps((random.getstate(), np.random.get_state())), torch.get_rng_state().numpy().tobytes()


def pair_names(pair):
    return tuple(None if entity is None else entity.name for entity in pair)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


@pytest.mark.parametrize('kind', ['mixed', 'reincarnator-golden', 'outside', 'plain', 'titans', 'ordinary'])
def test_native_selections_and_every_rng_continuation_match_frozen_order(native, kind):
    manager, population, reincarnator = make_case(kind, 9)
    other, expected_population, expected_reincarnator = make_case(kind, 9)
    index = candidate().try_create_aeonic_index(manager, population, reincarnator)
    assert index is not None, 'A standard nontrivial population must use the native index'
    seed(731)
    try:
        for step in range(80):
            before = random.getstate()
            expected = frozen()['select_aeonic_opponents'](other, expected_population, expected_reincarnator)
            expected_rng = rng_state()
            random.setstate(before)
            actual = index.select()
            assert pair_names(actual) == pair_names(expected)
            assert rng_state() == expected_rng
            if kind == 'ordinary':
                frozen()['aeonic_interaction'](other, *expected, None, expected_population, np.zeros(12))
                manager.aeonic_interaction(*actual, None, population, np.zeros(12))
                index.update_pair(*actual)
            elif step % 11 == 0:
                for rows in (population, expected_population):
                    rows[-2].trait = 'GoldenOne' if step % 22 else 'Mortal'
                    rows[-2].is_titan_form = TITAN_NAMES[4] if step % 22 else ''
                    rows[-1].held_fire_seeds = {TITAN_NAMES[6]} if step % 22 else set()
                assert index.refresh() is True
    finally:
        index.close()
        index.close()
        manager.population_manager.close_native_world()
        other.population_manager.close_native_world()
    with pytest.raises(RuntimeError, match='closed'):
        index.select()


@pytest.mark.parametrize('size', [0, 1])
def test_empty_and_singleton_keep_original_draw_rules(native, size):
    manager, population, reincarnator = make_case('mixed', size)
    index = candidate().try_create_aeonic_index(manager, population, reincarnator)
    seed(19)
    before = random.getstate()
    expected = frozen()['select_aeonic_opponents'](manager, population, reincarnator)
    expected_rng = rng_state()
    random.setstate(before)
    actual = index.select() if index is not None else manager.select_aeonic_opponents(population, reincarnator)
    assert actual == expected and rng_state() == expected_rng
    if index is not None:
        index.close()


@pytest.mark.parametrize('unsafe', ['duplicate', 'shared_seeds', 'defeated_alias', 'seed_object'])
def test_unsafe_metadata_returns_none_without_rng_or_seed_mutation(native, unsafe):
    manager, population, reincarnator = make_case()
    if unsafe == 'duplicate':
        population.append(population[1])
    elif unsafe == 'shared_seeds':
        population[-1].held_fire_seeds = population[-2].held_fire_seeds
    elif unsafe == 'defeated_alias':
        population[-1].held_fire_seeds = manager.defeated_titans_this_cycle
    else:
        population[-1].held_fire_seeds.add(object())
    before = rng_state()
    fields = [(p, p.held_fire_seeds, p.held_fire_seeds.copy(), p.trait, p.is_titan_form) for p in population]
    assert candidate().try_create_aeonic_index(manager, population, reincarnator) is None
    assert rng_state() == before
    for p, original, values, trait, form in fields:
        assert p.held_fire_seeds is original and p.held_fire_seeds == values
        assert (p.trait, p.is_titan_form) == (trait, form)


def assert_same(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert_same(got, want)
    else:
        assert actual == expected


def snapshot(manager, population, reincarnator, culled):
    def entity(p):
        return None if p is None else {key: value for key, value in vars(p).items()
                                     if key != '_titan_to_path_model'}
    return copy.deepcopy({
        'population': [entity(p) for p in population], 'reincarnator': entity(reincarnator),
        'culled': sorted(p.name for p in culled), 'names': manager.existing_names,
        'mapping': {name: p.name for name, p in manager.name_to_entity_map.items()},
        'defeated': manager.defeated_titans_this_cycle,
        'destruction': manager.baie_destruction_score, 'collected': manager.baie_total_seeds_collected,
        'actor': manager.action_policy_network.state_dict(), 'value': manager.value_network.state_dict(),
        'actor_optimizer': manager.action_optimizer.state_dict(), 'value_optimizer': manager.value_optimizer.state_dict(),
        'rng': rng_state(),
    })


def attach_hook(manager, population, mode):
    fired = []
    if mode is None:
        return None, fired

    def callback(module, inputs, output):
        if fired:
            return
        fired.append(True)
        population[-1].trait = 'GoldenOne'
        population[-1].is_titan_form = TITAN_NAMES[7]
        population[-1].held_fire_seeds.add(TITAN_NAMES[8])
        if mode == 'membership':
            population.reverse()
            population.pop()
            entity = Pathstrider('callback-newborn', np.ones(12), 'GoldenOne',
                                titan_to_path_model=manager.titan_to_path_model_instance)
            entity.score = 10.
            population.append(entity)
            manager.name_to_entity_map[entity.name] = entity
            manager.existing_names.add(entity.name)
        elif mode == 'invalidates':
            population.append(population[-1])
        elif mode == 'raises':
            raise RuntimeError('callback stopped the interaction')

    return manager.action_policy_network.register_forward_hook(callback), fired


class SpyIndex:
    def __init__(self, target):
        self.target, self.selections, self.pairs, self.refreshes = target, [], [], []
        self.closed = 0
        self.valid = True

    def select(self):
        assert self.valid and not self.closed, 'A closed or invalidated index must not select again'
        result = self.target.select()
        self.selections.append((pair_names(result), rng_state()))
        return result

    def update_pair(self, attacker, defender):
        self.pairs.append(pair_names((attacker, defender)))
        return self.target.update_pair(attacker, defender)

    def refresh(self):
        self.valid = self.target.refresh()
        self.refreshes.append(self.valid)
        return self.valid

    def close(self):
        self.closed += 1
        return self.target.close()


def run_case(kind, hook=None, reference=False, random_seed=731, shared_seeds=False):
    manager, population, reincarnator = make_case(kind)
    if shared_seeds:
        population[-1].held_fire_seeds = population[-2].held_fire_seeds
    handle, fired = attach_hook(manager, population, hook)
    traces, culled_seen = [], set()
    if reference:
        def select(self, rows, reinc):
            pair = frozen()['select_aeonic_opponents'](self, rows, reinc)
            traces.append((pair_names(pair), rng_state(), any(p in culled_seen for p in pair if p is not None)))
            return pair

        def interact(self, *arguments):
            result = frozen()['aeonic_interaction'](self, *arguments)
            if result is not None:
                culled_seen.add(result)
            return result

        manager.select_aeonic_opponents = MethodType(select, manager)
        manager.aeonic_interaction = MethodType(interact, manager)
    seed(random_seed)
    error, culled = None, set()
    try:
        function = frozen()['run_aeonic_cycle_generation'] if reference else AeonicCycleManager.run_aeonic_cycle_generation
        try:
            culled = function(manager, population, reincarnator, np.zeros(12))
        except RuntimeError as failure:
            if hook != 'raises':
                raise
            error = str(failure)
        result = snapshot(manager, population, reincarnator, culled)
        return result, traces, bool(fired), error
    finally:
        if handle is not None:
            handle.remove()
        manager.population_manager.close_native_world()


def spy_factory(monkeypatch):
    bridge = candidate()
    original, spies, declines = bridge.try_create_aeonic_index, [], []

    def create(*arguments):
        result = original(*arguments)
        if result is None:
            declines.append(True)
            return None
        observed = SpyIndex(result)
        spies.append(observed)
        return observed

    monkeypatch.setattr(bridge, 'try_create_aeonic_index', create)
    if hasattr(cycle_module, 'try_create_aeonic_index'):
        monkeypatch.setattr(cycle_module, 'try_create_aeonic_index', create)
    return spies, declines


def test_generation_uses_pair_updates_and_keeps_culled_entities_in_draws(native, monkeypatch):
    # Find a deterministic frozen trajectory that actually redraws a dead
    # member, so an implementation which eagerly removes it cannot pass vacuously.
    for chosen_seed in range(32):
        expected, traces, _, _ = run_case('ordinary', reference=True, random_seed=chosen_seed)
        if any(row[2] for row in traces):
            break
    else:
        pytest.fail('The ordinary fixture did not exercise a culled-member draw')
    spies, declines = spy_factory(monkeypatch)
    actual, _, _, _ = run_case('ordinary', random_seed=chosen_seed)
    assert_same(actual, expected)
    assert not declines and len(spies) == 1
    index = spies[0]
    assert index.selections == [row[:2] for row in traces]
    assert index.pairs and not index.refreshes and index.closed
    assert len(index.selections) == 12


@pytest.mark.parametrize('hook', ['third_party', 'membership', 'invalidates', 'raises'])
def test_real_reincarnator_callbacks_refresh_all_members_or_fall_back_and_close(native, monkeypatch, hook):
    expected, traces, fired, expected_error = run_case('reincarnator-only', hook, reference=True)
    assert fired, 'The fixture must execute the real network callback'
    spies, declines = spy_factory(monkeypatch)
    actual, _, actual_fired, actual_error = run_case('reincarnator-only', hook)
    assert actual_fired and actual_error == expected_error
    assert_same(actual, expected)
    assert not declines and len(spies) == 1
    index = spies[0]
    assert index.selections == [row[:2] for row in traces[:len(index.selections)]]
    assert index.closed
    if hook == 'raises':
        assert expected_error == 'callback stopped the interaction' and not index.refreshes
    elif hook == 'invalidates':
        assert index.refreshes == [False]
    else:
        assert index.refreshes and all(index.refreshes)
        assert len(index.selections) == len(traces) == 12


def test_shared_seed_sets_keep_the_original_whole_generation_fallback(native, monkeypatch):
    expected, _, _, _ = run_case('ordinary', reference=True, shared_seeds=True)
    spies, declines = spy_factory(monkeypatch)
    actual, _, _, _ = run_case('ordinary', shared_seeds=True)
    assert_same(actual, expected)
    assert declines and not spies
