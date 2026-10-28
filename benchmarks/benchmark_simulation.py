"""Bounded human-phase benchmark; includes Python/native boundary and data movement."""
import argparse
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--population', type=int, default=1000)
parser.add_argument('--phase', choices=['human', 'inorganic'], default='human')
parser.add_argument('--generations', type=int, default=2)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--backend', choices=['python', 'cpp'], default='cpp')
parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root))
import numpy as np
import torch
import types
from config import config
config['llm']['enable_llm'] = False
config.setdefault('runtime', {}).update(torch_device='cpu', interactive=False)
if (args.source_root/'native_backend.py').exists():
    from native_backend import configure_backend
    configure_backend(args.backend)

# Exclude keyboard setup for original and new code equally. The benchmark calls
# actual generation methods, not main/start, and never exports policy files.
class QuietDebugger:
    def __init__(self, *a, **k):
        self.paused = False
        self.last_command = ''
        self.stop_requested = False
    def close(self): pass
    def poll(self): pass
stub = types.ModuleType('debugger')
stub.Debugger = QuietDebugger
sys.modules['debugger'] = stub
from simulation import AeonEvolution
logging.disable(logging.CRITICAL)
torch.set_num_threads(1)

def sample(population):
    random.seed(731)
    np.random.seed(731)
    torch.manual_seed(731)
    kw = {k:v for k,v in config['simulation'].items() if k!='elites_to_keep_in_cycle'}
    kw.update(num_initial_entities=population, population_soft_cap=population,
              population_hard_cap=population*2, bard_frequency=0, laertes_frequency=0,
              kaoselanna_llm_enabled=False)
    sim = AeonEvolution(**kw)
    sim.display_manager.update_and_display_progress = lambda *a, **k: None
    if args.phase == 'inorganic':
        start = time.perf_counter()
        legacy = sim._run_inorganic_phase(args.generations)
        seconds = time.perf_counter() - start
        sim.debugger.close()
        return seconds, legacy
    sim._create_initial_population(create_reincarnator=False)
    sim.legacy_manager.newborn_affinity_modifier = np.zeros(12)
    sim.generation = 176200
    sim.total_generations = 335503360
    start = time.perf_counter()
    for _ in range(args.generations):
        culled = sim._run_one_generation()
        sim._evolve_and_grow(culled)
        sim.generation += 1
    seconds = time.perf_counter()-start
    sim.debugger.close()
    return seconds, len(sim.population)

sample(20)
runs = [sample(args.population) for _ in range(args.repeats)]
result = {'backend':args.backend, 'source_root':str(args.source_root), 'initial_population':args.population,
          'phase':args.phase,
          'generations':args.generations, 'seconds':[x[0] for x in runs],
          'median_seconds':statistics.median(x[0] for x in runs), 'outcomes':[x[1] for x in runs],
          'python':sys.version.split()[0], 'numpy':np.__version__, 'torch':torch.__version__,
          'scope':'CPU 1 thread; seed 731; no LLM/log/keyboard; excludes startup; human case: neutral legacy, no reincarnator, excludes pre-phases; inorganic case: original 10x10 grid'}
text=json.dumps(result,indent=2)
print(text)
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text+'\n')
