"""Bounded real-phase replay, also executable against the frozen upstream tree.

Generate a reference fixture by passing --root to the unchanged 691c6e0 checkout.
Only console/display/export side effects are replaced; world computation, neural
forward/backward calls, optimizer steps, and random draws run unchanged.
"""
import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import logging
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch


BASELINE_COMMIT = "691c6e0023a83513ad3cd386cfee2198317104d3"
SEED = 731


class QuietDebugger:
    def __init__(self, *args, **kwargs):
        self.paused = False
        self.stop_requested = False
        self.last_command = ""

    def poll(self):
        pass

    def close(self):
        pass


def prohibit_export(*args, **kwargs):
    raise AssertionError("A bounded phase replay must not export policy files")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def entity_state(entity):
    # Include derived values and the lazy path cache, omitted by to_dict().
    # The shared mapping model is configuration, not per-entity mutable state.
    return {key: json_value(value) for key, value in vars(entity).items()
            if key != "_titan_to_path_model"}


def rng_state():
    def digest(data):
        return hashlib.sha256(data).hexdigest()
    numpy_state = np.random.get_state()
    numpy_payload = [numpy_state[0], numpy_state[1].tolist(),
                     int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])]
    return {
        "python": digest(repr(random.getstate()).encode()),
        "numpy": digest(json.dumps(numpy_payload, separators=(",", ":")).encode()),
        "torch": digest(torch.get_rng_state().numpy().tobytes()),
    }


def evaluate(model, probes):
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(probes).cpu().tolist()
    finally:
        model.train(training)


def build(args):
    sys.path.insert(0, str(Path(args.root).resolve()))
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from config import config
    config["llm"]["enable_llm"] = False
    config["llm"]["kaoselanna_llm_enabled"] = False
    config["log"]["enable"] = False
    config.setdefault("runtime", {}).update(torch_device="cpu", cpu_threads=1, interactive=False)
    if (Path(args.root) / "native_backend.py").exists():
        from native_backend import configure_backend
        configure_backend(args.backend)
    console = types.ModuleType("debugger")
    console.Debugger = QuietDebugger
    sys.modules["debugger"] = console
    from simulation import AeonEvolution
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    options = {key: value for key, value in config["simulation"].items()
               if key != "elites_to_keep_in_cycle"}
    options.update(num_initial_entities=args.population, population_soft_cap=args.population,
                   population_hard_cap=2 * args.population, bard_frequency=0,
                   laertes_frequency=0, kaoselanna_llm_enabled=False)
    sim = AeonEvolution(**options)
    sim.display_manager.update_and_display_progress = lambda *args, **kwargs: None
    sim.policy_saver.save_policy_models = prohibit_export
    sim.policy_saver.save_organic_model = prohibit_export
    return sim, config


def inorganic(sim):
    return {"generations": 17, "legacy": json_value(sim._run_inorganic_phase(17)), "rng": rng_state()}


def aeonic(sim, config):
    from constants import TITAN_NAMES
    sim._create_initial_population(create_reincarnator=False)
    sim.population_manager.update_golden_ones(sim.population)
    sim.reincarnator = next((p for p in sim.population if p.titan_aspect == "负世"),
                           max(sim.population, key=lambda p: p.score))
    sim.reincarnator.trait = "Reincarnator"
    sim.reincarnator.name = "Neikos-0496"
    sim.aeonic_cycle_mode = True
    sim.generation = config["simulation_phases"]["HUMAN_PHASE_END"]
    manager = sim.aeonic_cycle_manager
    assert manager.initialize_aeonic_cycle(sim.reincarnator, sim.population, sim.cosmic_zeitgeist)
    assert len(manager.titan_entities) == len(TITAN_NAMES) == 12

    # Fixed probes consume no random draws. The final row is a real initial
    # attacker/defender state from this population, alongside three fixed rows.
    real_state = np.concatenate([sim.reincarnator.titan_affinities,
                                 manager.titan_entities[TITAN_NAMES[0]].titan_affinities])
    probes = torch.stack([torch.zeros(24), torch.ones(24),
                          torch.linspace(-1, 1, 24), torch.from_numpy(real_state).float()])
    networks = {"action": (sim.action_policy_network, sim.action_optimizer),
                "value": (sim.value_network, sim.value_optimizer)}
    initial_parameters = {}
    network_states = {}
    for name, (model, optimizer) in networks.items():
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        initial_parameters[name] = [parameter.detach().clone() for parameter in model.parameters()]
        network_states[name] = {"before": evaluate(model, probes)}

    outcomes = []
    for generation in range(2):
        if generation:
            # Preserve the real outer-loop promotion order before generation 2.
            last_titans = [p for p in sim.population if p.titan_aspect and p.trait == "GoldenOne"]
            sim.titan_bosses.clear()
            for entity in last_titans:
                entity.is_titan_boss = True
                entity.trait = "Mortal"
                entity.hp = 250.0 + entity.score
                sim.titan_bosses.append(entity)
            sim.population_manager.update_golden_ones(sim.population)
        before = len(sim.population)
        culled = manager.run_aeonic_cycle_generation(sim.population, sim.reincarnator, sim.cosmic_zeitgeist)
        cycle_finished = len(sim.reincarnator.held_fire_seeds) >= len(TITAN_NAMES)
        should_end = False
        if cycle_finished:
            should_end = manager.end_aeonic_cycle(
                sim.reincarnator, sim.population, sim.base_titan_affinities,
                sim.population_manager.mutation_rate, sim.population_soft_cap,
                sim.cosmic_zeitgeist, config["simulation"]["elites_to_keep_in_cycle"])
        outcomes.append({"population_before": before, "population_after": len(sim.population),
                         "culled_names": sorted(entity.name for entity in culled),
                         "held_fire_seeds": sorted(sim.reincarnator.held_fire_seeds),
                         "cycle_finished": cycle_finished, "should_end": bool(should_end),
                         "reincarnator_in_population": sim.reincarnator in sim.population,
                         "active_titans": sum(bool(entity.is_titan_form) for entity in sim.population)})
        sim.generation += 1
        if should_end:
            break

    for name, (model, optimizer) in networks.items():
        initial = initial_parameters[name]
        changed = sum(not torch.equal(old, new.detach()) for old, new in zip(initial, model.parameters()))
        maximum_change = max((old - new.detach()).abs().max().item()
                             for old, new in zip(initial, model.parameters()))
        steps = [int(optimizer.state[parameter]["step"].item()) for parameter in model.parameters()]
        # A forward-only or skipped-training implementation cannot pass this.
        assert changed > 0 and maximum_change > 0 and min(steps) > 0, name
        network_states[name].update(after=evaluate(model, probes), changed_tensors=changed,
                                    maximum_parameter_change=maximum_change, optimizer_steps=steps)

    return {
        "generation": sim.generation, "outcomes": outcomes,
        "population_order": [entity.name for entity in sim.population],
        "population": [entity_state(entity) for entity in sorted(sim.population, key=lambda p: p.name)],
        "reincarnator": entity_state(sim.reincarnator),
        "titan_entities": {name: entity_state(entity) for name, entity in manager.titan_entities.items()},
        "titan_bosses": [entity_state(entity) for entity in sim.titan_bosses],
        "name_to_entity_map": {name: entity.name for name, entity in sim.name_to_entity_map.items()},
        "existing_names": sorted(sim.existing_names),
        "blueprint": sim.base_titan_affinities.tolist(),
        "cosmic_zeitgeist": sim.cosmic_zeitgeist.tolist(),
        "seats": json_value(sim.parliament_manager.seats),
        "cycle": {"number": manager.aeonic_cycle_number,
                  "defeated_titans": sorted(manager.defeated_titans_this_cycle),
                  "destruction_score": float(manager.baie_destruction_score),
                  "total_seeds_collected": manager.baie_total_seeds_collected},
        "network_probe_inputs": probes.tolist(), "networks": network_states, "rng": rng_state(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--phase", choices=("inorganic", "aeonic"), required=True)
    parser.add_argument("--population", type=int, default=100)
    args = parser.parse_args()
    sim, config = build(args)
    result = inorganic(sim) if args.phase == "inorganic" else aeonic(sim, config)
    result.update(baseline_commit=BASELINE_COMMIT, seed=SEED, phase=args.phase,
                  torch_threads=torch.get_num_threads())
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    if hasattr(sim, 'close'):
        sim.close()
    else:
        sim.debugger.close()


if __name__ == "__main__":
    main()
