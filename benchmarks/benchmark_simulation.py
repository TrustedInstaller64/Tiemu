"""Bounded phase timing; includes native boundaries and actual phase methods.

Use on hosted CI while local high-load execution is prohibited. Inorganic
results include the complete RNG-state hashes; the Python hash includes its
cached Gaussian, even though this phase consumes only uniform random values.
"""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import json
import logging
import pickle
import platform
import random
import statistics
import subprocess
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
if min(args.population, args.generations, args.repeats) < 1:
    parser.error('population, generations and repeats must be positive')
root = args.source_root.resolve()
sys.path.insert(0, str(root))
import numpy as np
import torch
import types
from config import config
config['llm']['enable_llm'] = False
config['log']['enable'] = False
config.setdefault('runtime', {}).update(torch_device='cpu', interactive=False,
                                       native_threads=1, native_min_rows=4096)
native = None
if (root/'native_backend.py').exists():
    from native_backend import configure_backend
    native = configure_backend(args.backend)
elif args.backend == 'cpp':
    raise RuntimeError('The selected source root does not provide the requested C++ backend')
core_file = None
if native is not None:
    core_file = str(Path(getattr(native, '__core_file__', native.__file__)).resolve())
    for location in (native.__file__, core_file):
        if Path(location).resolve().parent != root:
            raise RuntimeError('Build the native library in-place in each source root before comparing')

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
if Path(sys.modules['simulation'].__file__).resolve().parent != root:
    raise RuntimeError('The benchmark imported simulation from the wrong source root')
logging.disable(logging.CRITICAL)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def inorganic_state(sim, legacy):
    # Serialize the actual states rather than array reprs, which may truncate
    # under NumPy print options. protocol=4 is identical within the paired runs.
    def state_hash(state):
        return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()
    checked = {
        'legacy': {key: float(value) for key, value in legacy.items()},
        'generation': sim.generation,
        'python_rng_sha256': state_hash(random.getstate()),
        'numpy_rng_sha256': state_hash(np.random.get_state()),
        'torch_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }
    checked['state_digest'] = hashlib.sha256(json.dumps(
        checked, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return checked

def sample(population):
    random.seed(731)
    np.random.seed(731)
    torch.manual_seed(731)
    kw = {k:v for k,v in config['simulation'].items() if k!='elites_to_keep_in_cycle'}
    kw.update(num_initial_entities=population, population_soft_cap=population,
              population_hard_cap=population*2, bard_frequency=0, laertes_frequency=0,
              kaoselanna_llm_enabled=False)
    sim = AeonEvolution(**kw)
    try:
        sim.display_manager.update_and_display_progress = lambda *a, **k: None
        if args.phase == 'inorganic':
            start_cpu, start = time.process_time(), time.perf_counter()
            legacy = sim._run_inorganic_phase(args.generations)
            seconds = time.perf_counter() - start
            cpu_seconds = time.process_time() - start_cpu
            if legacy is None or sim.generation != args.generations:
                raise AssertionError('The inorganic phase did not finish all requested generations')
            return {'seconds': seconds, 'cpu_seconds': cpu_seconds, 'outcome': legacy,
                    'checked_state': inorganic_state(sim, legacy)}
        sim._create_initial_population(create_reincarnator=False)
        sim.legacy_manager.newborn_affinity_modifier = np.zeros(12)
        sim.generation = 176200
        sim.total_generations = 335503360
        start_cpu, start = time.process_time(), time.perf_counter()
        for _ in range(args.generations):
            culled = sim._run_one_generation()
            sim._evolve_and_grow(culled)
            sim.generation += 1
        seconds = time.perf_counter() - start
        return {'seconds': seconds, 'cpu_seconds': time.process_time() - start_cpu,
                'outcome': len(sim.population)}
    finally:
        if hasattr(sim, 'close'):
            sim.close()
        else:
            sim.debugger.close()

sample(20)
runs = [sample(args.population) for _ in range(args.repeats)]
result = {'backend':args.backend, 'source_root':str(root), 'initial_population':args.population,
          'phase':args.phase,
          'generations':args.generations, 'seconds':[x['seconds'] for x in runs],
          'median_seconds':statistics.median(x['seconds'] for x in runs),
          'outcomes':[x['outcome'] for x in runs],
          'python':sys.version.split()[0], 'numpy':np.__version__, 'torch':torch.__version__,
          'platform':platform.platform(), 'machine':platform.machine(), 'logical_cpus':os.cpu_count(),
          'commit':subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
          'core_file':core_file, 'native_binding':getattr(native, '__binding__', 'legacy-extension') if native else None,
          'cpu_seconds':[x['cpu_seconds'] for x in runs],
          'scope':'CPU/BLAS/PyTorch 1 thread; seed 731; no LLM/log/display/keyboard/exports. '
                  'Warm-up and measured samples use separate simulations. Process/import/simulation '
                  'setup, warm-up, final state hashing and close are excluded. Actual phase calls '
                  'and their Python/native data movement are timed. Human: neutral legacy, no '
                  'reincarnator, excludes pre-phases. Inorganic: original 10x10 grid including its '
                  'initial random cell values and final legacy reduction; new C ABI headless loop '
                  'checks controls every 100 steps, whereas 4a2311d checks every generation. '
                  'Display is stubbed but differences in presentation/control call overhead remain. '
                  'Not a pure-kernel, full-world, LLM or local-M1 performance claim.'}
if args.phase == 'inorganic':
    result['checked_states'] = [x['checked_state'] for x in runs]
    result['state_scope'] = ('Legacy, final generation, full Python MT state including Gaussian cache, '
                             'full NumPy RNG state and Torch CPU RNG. The phase does not expose its '
                             'final cell arrays; complete array/RNG equivalence is covered by engine tests.')
text=json.dumps(result,indent=2,allow_nan=False)
print(text)
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text+'\n', encoding='utf-8')
