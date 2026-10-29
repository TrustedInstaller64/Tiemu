"""Reproduce a bounded world trajectory; usable against the frozen upstream tree."""
import argparse
import hashlib
import json
import logging
import random
import sys
import types
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--backend', default='python')
parser.add_argument('--reincarnator', action='store_true')
parser.add_argument('--population', type=int, default=40)
parser.add_argument('--generations', type=int, default=3)
parser.add_argument('--native-threads', type=int, default=1)
parser.add_argument('--native-min-rows', type=int, default=4096)
args = parser.parse_args()
sys.path.insert(0, args.root)
import numpy as np
import torch
logging.disable(logging.CRITICAL)
torch.set_num_threads(1)
from config import config
config['llm']['enable_llm'] = False
config.setdefault('runtime', {})['torch_device'] = 'cpu'
config['runtime']['interactive'] = False
config['runtime'].update(native_threads=args.native_threads, native_min_rows=args.native_min_rows)
if (Path(args.root) / 'native_backend.py').exists():
    from native_backend import configure_backend
    configure_backend(args.backend)

class QuietDebugger:
    def __init__(self, *a, **k):
        self.paused = False
        self.last_command = ''
        self.stop_requested = False
    def poll(self): pass
    def close(self): pass

stub = types.ModuleType('debugger')
stub.Debugger = QuietDebugger
sys.modules['debugger'] = stub
from simulation import AeonEvolution
random.seed(731)
np.random.seed(731)
torch.manual_seed(731)
kw = {k: v for k, v in config['simulation'].items() if k != 'elites_to_keep_in_cycle'}
kw.update(num_initial_entities=args.population, population_soft_cap=args.population,
          population_hard_cap=2*args.population, bard_frequency=0, laertes_frequency=0,
          kaoselanna_llm_enabled=False)
sim = AeonEvolution(**kw)
sim._create_initial_population(create_reincarnator=args.reincarnator)
sim.legacy_manager.newborn_affinity_modifier = np.zeros(12)
sim.generation = 176200
sim.total_generations = 335503360
for _ in range(args.generations):
    culled = sim._run_one_generation()
    sim._evolve_and_grow(culled)
    sim.generation += 1

def digest(data):
    return hashlib.sha256(data).hexdigest()

result = {
    'population': [p.to_dict() for p in sorted(sim.population, key=lambda p: p.name)],
    'blueprint': sim.base_titan_affinities.tolist(),
    'seats': {k: int(v) for k, v in sim.parliament_manager.seats.items()},
    'random_state': digest(repr(random.getstate()).encode()),
    'numpy_state': digest(repr(np.random.get_state()).encode()),
    'torch_state': digest(torch.get_rng_state().numpy().tobytes()),
}
print(json.dumps(result, ensure_ascii=True, sort_keys=True))
if hasattr(sim, 'close'):
    sim.close()
else:
    sim.debugger.close()
