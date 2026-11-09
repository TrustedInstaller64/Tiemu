# Tiemu

Tiemu is an evolution simulator set in the Amphoreus universe of *Honkai: Star Rail*. A full run goes through an inorganic 10×10 grid, an organic prototype network, a human civilization with elections and Titans, and finally the aeonic recurrence, where a reincarnator collects the twelve Titan seeds and keeps training online.

This repository starts from [YezQiu/Tiemu](https://github.com/YezQiu/Tiemu) at `691c6e0` and keeps its world rules. Most of the work is a standalone C++17 core for the measured hot paths, plus cross-platform builds and a differential test suite. v1.0.0 is the strict-equivalence A line: states, random streams and every optimizer update match the reference. The separate C experiments, which allow trajectories to diverge, are not included.

[中文说明](readme/readme_zh-cn.md) · [Release notes](RELEASE_NOTES.md) · [Performance records](docs/performance.md) · [Parameter guide](readme/Guide.md) · [Logging](readme/clog_en.md)

## Install

Python 3.11–3.13 (3.12 is what gets tested most) and a C++17 compiler. NumPy and PyTorch are pinned to 2.2.2 and 2.12.0.

```sh
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[test]"
python -m main --backend cpp --device cpu --disable-llm --smoke-test
```

`pip install ".[test]"` is what compiles `_tiemu_core`. If you only install `requirements.txt`, you get the Python backend; either use `--backend python` or build the core separately:

```sh
cmake -S . -B build/standalone -DCMAKE_BUILD_TYPE=Release
cmake --build build/standalone --config Release
ctest --test-dir build/standalone -C Release --output-on-failure
```

After changing anything under `cpp/`, reinstall the project. The old shared library does not refresh itself.

Drop `--smoke-test` for the real phase lengths. Output goes to `runs/` unless `--output-dir` says otherwise.

## How it is put together

Python runs the show: CLI and config, phase order, the four PyTorch networks and their online training, lifecycle decisions, the console and the snapshots. The C++ core does the work that showed up in profiles: path arithmetic, ordered candidate indexing, histograms, the scoring worker pool, inorganic grid segments, ordered encounters, the resident world state and batched births.

The boundary is a C ABI. `_tiemu_ffi.py` loads `_tiemu_core` with `ctypes`. There is no pybind11, and the core does not include Python or NumPy headers.

`--backend auto` uses the native core when it is there and falls back with a warning. `--backend cpp` refuses to start without it. `--backend python` keeps the reference path.

## Threads

`--device` picks the device for the small networks. CPU with one PyTorch thread is the default; on the reference M1 that was faster than MPS for these single-sample forwards.

`--native-threads` is the budget for independent entity scoring. It includes the calling thread and only starts workers when at least `--native-min-rows` entities are being scored (default 4096). The default is one thread, and four threads did not produce a stable whole-phase gain. Encounters, random draws, vote reduction and optimizer updates stay in order.

`--compute-only` cuts logging and display waits without skipping simulation work or changing random-number consumption.

## Pausing and output

Press `p` at a safe point. The usual commands are `c`, `n`, `status`, `top`, `p <name|baie>`, `zeitgeist`, `blueprint`, `set <param> <value>`, `save <file.json>`, `stop` and `help`. Without a TTY, keyboard input turns itself off.

Normal completion, `stop`, Ctrl+C and SIGTERM write `world.json` and the three policy weights to the output directory, using atomic replacement. `organic_prototype_model.pth` is written if organic training finished.

`world.json` is a snapshot to look at, not a checkpoint. It has no optimizer, RNG or phase-internal state, so `load` and `--load-save` refuse to restore it instead of restarting or overwriting a world.

## LLM (optional)

No model is downloaded for you. `--disable-llm` turns narration off entirely.

On Apple Silicon, use `--llm-backend mlx --mlx-model` with a local MLX directory. Elsewhere, use `--llm-backend llama_cpp --gguf-model` with a local GGUF file; CUDA builds of llama.cpp work as well. `auto` tries MLX on macOS arm64 first, then GGUF, and disables the LLM with a logged reason if neither loads. Sampling and quantization change trajectories, so don't expect identical text across backends.

## Docker

```sh
docker compose build
docker compose up -d
docker compose logs -f
docker compose stop
```

The image is CPU-only, runs as UID 10001, starts with the LLM disabled, and keeps output in the `tiemu-data` volume. `docker build --build-arg WITH_LLM=1` adds CPU GGUF support. GPU passthrough is not configured.

## Performance

Paired, fixed-phase samples. They are not whole-program numbers, and ratios from different versions or workloads should not be multiplied.

| Workload | Comparison | Result |
|---|---|---|
| Human 1,000×6 | A2 vs A1 | 2.10–2.30× |
| Human 10,000×3 | A2 vs A1 | 1.32–1.47× |
| Inorganic 10×10×10,000 | A2 vs A1 | 9.13–13.53× |
| Aeonic 200×2 (real training) | A2 vs A1 | 0.944–0.958× (slower) |
| Human 10,000×3 | A4 vs A3 | 1.068× Linux / 1.140× Windows / 1.041× hosted macOS |
| Aeonic 1,000×12 (full cycle) | A4 vs A3 | 1.438× / 1.610× / 1.506× |

Some cases are worse than the baseline: A2 aeonic 200×2 at 0.944–0.958×, A3 macOS aeonic 200×2 at 0.934×, A4 macOS aeonic 200×2/200×12 at 0.908/0.902×, and A4 macOS inorganic at 0.751×. The 1,000×12 aeonic sample only runs 44 optimizer updates per optimizer because the original loop loses the reincarnator reference after a host change; it is not a sustained-training result. Four scoring threads show no stable whole-phase gain, so the default stays at one.

The 31.65× human and 6.23× inorganic numbers from 2026-10-28 are stage measurements on a physical M1 against the original Python loop, kept for reference.

Full tables and raw evidence: [docs/performance.md](docs/performance.md).

## Tests

196 test functions. In the A5 baseline, the parametrized suite passed 613 tests on Linux/macOS and 601 with 12 platform skips on Windows. The differential tests cover arithmetic, ordered encounters, scoring, births, lifetimes, the resident world and the C ABI. Discrete decisions and RNG states are compared exactly; numeric tolerances are `1e-10` for world values and about `1e-6` for float32 networks.

The equivalence tests replay upstream `691c6e0` from Git history, so a shallow clone needs `git fetch --unshallow` first.

## Layout

```
main.py, simulation.py     CLI and main engine
*_manager.py               parliament, diversity, population, stagnation, aeonic cycle
entities.py, models.py     entities and networks
native_*.py, _tiemu_ffi.py native adapters and the C ABI bridge
cpp/                       C++17 core
tests/                     pytest suite, C consumer, frozen fixtures
benchmarks/                timing and comparison tools
docs/                      performance records and design docs
```

## Thanks

Original project and contributors: YezQiu, HeyCrab3, AvalonRuFae, MoonofBridge24, Akanyi. Logging uses [pyclog](https://github.com/Akanyi/pyclog).

MIT licensed. This is an unofficial fan project; the game setting belongs to its owners.
