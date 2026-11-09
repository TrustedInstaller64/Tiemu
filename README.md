# Tiemu

Tiemu is an evolution simulator inspired by the Amphoreus setting of *Honkai: Star Rail*. A run moves through four phases: an inorganic 10×10 grid, an organic prototype network, a human/Pathstrider civilization with elections and Titans, and an aeonic recurrence with online reinforcement learning.

This repository is based on [YezQiu/Tiemu](https://github.com/YezQiu/Tiemu) at commit `691c6e0`. The original world rules are preserved; the main additions are a standalone C++17 numerical core, cross-platform packaging and differential validation.

[中文说明](readme/readme_zh-cn.md) · [Release notes](RELEASE_NOTES.md) · [Performance and equivalence](docs/performance.md) · [World parameter guide](readme/Guide.md) · [Logging](readme/clog_en.md)

## v1.0.0 at a glance

- Python 3.11–3.13 orchestration layer with an independent C++17 core.
- The two layers communicate through a C ABI and a thin `ctypes` adapter; pybind11 is not used.
- A resident C++ World owns continuous numeric state, stable entity IDs, population order, membership reconciliation and batched birth arithmetic.
- Ordered semantics are preserved: encounter order, random-number consumption, old cache behavior and every online optimizer update.
- Linux, Windows and macOS CI; standalone CMake/CTest; Linux CPU Docker image.
- 196 test functions, including frozen-upstream replays and strict RNG/discrete comparisons.

## Architecture

Tiemu is a Python application with a native numerical core. It is not a pure C++ rewrite and does not embed Python.

| Layer | Responsibilities |
|---|---|
| Python | CLI and configuration, phase orchestration, PyTorch model definitions and online training, lifecycle decisions, console control, snapshots, optional LLM backends |
| C++17 core (`cpp/`) | Numerical kernels, ordered candidate indexing, histograms, the scoring worker pool, inorganic grid segments, ordered entity encounters, resident world state, batched birth arithmetic |
| Bridge | C ABI shared library `_tiemu_core`, loaded by the thin adapter `_tiemu_ffi` through `ctypes`; no Python C API, NumPy headers or pybind11 |

Backend selection:

- `--backend auto` (default): use the native core when it is available; otherwise log a warning and fall back to Python.
- `--backend cpp`: require the native core; fail clearly if it is missing.
- `--backend python`: keep the reference path for comparison and unusual inputs.

## Requirements

- Python 3.11–3.13 (3.12 recommended).
- A C++17 compiler: GCC on Linux, Xcode Command Line Tools on macOS, Visual Studio Build Tools with the C++ workload on Windows.
- Pinned numerical dependencies: NumPy `2.2.2`, PyTorch `2.12.0`.
- Optional LLM backends: MLX on Apple Silicon or `llama-cpp-python` with a local GGUF model. No model is downloaded automatically.

## Install

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[test]"
```

`pip install ".[test]"` compiles and installs the C++ shared core. Installing only `requirements.txt` does not build it; in that case use `--backend python` or build the core separately. After changing C++ sources, reinstall the project so the shared library is rebuilt.

The core can also be built without Python:

```sh
cmake -S . -B build/standalone -DCMAKE_BUILD_TYPE=Release
cmake --build build/standalone --config Release
ctest --test-dir build/standalone -C Release --output-on-failure
```

## Quick start

```sh
# Short end-to-end check (24 entities, shortened phases)
python -m main --backend cpp --device cpu --disable-llm --smoke-test

# Full simulation
python -m main --backend cpp --device cpu --torch-threads 1 --disable-llm --fast-forward --output-dir runs/cpu

# Test suite (requires full Git history for frozen-upstream replay)
python -m pytest -q
```

Remove `--smoke-test` for the full phase lengths. Outputs go to `runs/` by default; change this with `--output-dir`.

## Backends and threading

- `--device {auto,cpu,cuda,mps}` selects the device for the small online networks. CPU with one PyTorch thread is the default because it was fastest for these single-sample workloads on the reference M1.
- `--torch-threads N` sets PyTorch CPU threads.
- `--native-threads N` sets the C++ budget for independent entity scoring. The budget includes the calling thread; workers start only when at least `--native-min-rows` entities are scored. Defaults are `1` and `4096`.
- `--compute-only` suppresses INFO logging and display waits while preserving simulation work, random-number consumption and logging side effects.

Only independent scoring runs in parallel. Ordered encounters, random draws, vote reduction and online optimizer updates remain sequential.

## Console, outputs and snapshots

With a real terminal, press `p` at a safe point to pause. Commands include `c`, `n`, `status`, `top`, `p <name|baie>`, `zeitgeist`, `blueprint`, `set <parameter> <value>`, `save <path.json>`, `stop`, and `help`. Non-TTY input is disabled automatically; use `--no-interactive` to make this explicit.

Normal completion, `stop`, Ctrl+C and SIGTERM write `world.json` and CPU-portable policy weights with atomic replacement. When organic training completed, `organic_prototype_model.pth` is also written.

`world.json` is an inspection snapshot, not a resumable checkpoint. It omits optimizer state, random-number state and phase-internal data. `load` and `--load-save` therefore refuse restoration instead of silently restarting a world or overwriting existing output.

## Optional LLM backends

LLM narration is optional and disabled with `--disable-llm`.

- Apple Silicon: MLX/Metal with a local model directory (`--llm-backend mlx --mlx-model /path/to/Qwen3-0.6B-MLX`).
- Windows/Linux or CPU-only: `llama-cpp-python` with a local GGUF file (`--llm-backend llama_cpp --gguf-model /path/to/model.gguf`), optionally with CUDA.
- `auto` prefers MLX on native Apple Silicon macOS, then GGUF; if nothing is available, the LLM is disabled with a logged reason.

Models are never downloaded implicitly. Independently sampled or quantized LLMs do not produce identical long trajectories.

## Docker (CPU)

```sh
docker compose build
docker compose up -d
docker compose logs -f
docker compose stop
```

The default image runs as a non-root user (UID 10001), disables the LLM, and persists output in the `tiemu-data` volume. It does not require GPU passthrough. To build an image with CPU GGUF support:

```sh
docker build --build-arg WITH_LLM=1 -t tiemu:cpu-llm .
```

## Platform support

- Linux, Windows and macOS are exercised by the `native` CI workflow: standalone CMake/CTest, pure C ABI consumer, `pip install`, pytest and CLI smoke tests.
- Linux CPU Docker is built and smoke-tested in CI.
- Historical M1 measurements cover macOS C++/CPU, POSIX console, MLX Metal and MPS forwards.
- Windows console logic is tested against a simulated API. Windows/CUDA, Hyper-V GPU passthrough and ANE/Core ML are not validated on physical hardware. There is no Core ML backend.

## Performance summary

These are fixed-phase samples from paired runs, not whole-program or long-term throughput promises. Ratios from different versions or workloads must not be multiplied. Hosted macOS is a virtual M1, not physical hardware.

| Workload | Comparison | Result |
|---|---|---|
| Human 1,000×6 | A2 vs A1 | 2.10–2.30× |
| Human 10,000×3 | A2 vs A1 | 1.32–1.47× |
| Inorganic 10×10×10,000 | A2 vs A1 | 9.13–13.53× |
| Aeonic 200×2 (real training) | A2 vs A1 | 0.944–0.958× (slower) |
| Human 10,000×3 | A4 vs A3 | 1.068× Linux / 1.140× Windows / 1.041× hosted macOS |
| Aeonic 1,000×12 (full cycle) | A4 vs A3 | 1.438× Linux / 1.610× Windows / 1.506× hosted macOS |

Important limits:

- The A4/A5 `1,000×12` full-cycle sample performs only 44 optimizer updates per optimizer because the original loop loses the reincarnator reference after a host change. It is not a sustained-training speedup.
- Some workloads regress: A2 aeonic `200×2` at 0.944–0.958×; A3 macOS aeonic `200×2` at 0.934×; A4 macOS aeonic `200×2`/`200×12` at 0.908/0.902× and macOS inorganic at 0.751×.
- Four scoring threads show no stable whole-phase gain; the default remains one thread.
- The 2026-10-28 physical-M1 stage baselines (31.65× human and 6.23× inorganic versus the original Python loop) are historical stage measurements only.

Full tables, raw evidence and platform limits are in [docs/performance.md](docs/performance.md).

## Testing and equivalence

- 196 test functions. The parametrized default suite passes 613 tests on Linux/macOS and 601 with 12 platform skips on Windows (A5 baseline).
- Differential tests cover world arithmetic, ordered encounters, scoring, births, lifetimes, the resident World and the pure C ABI.
- Frozen upstream commit `691c6e0` is replayed on the same platform; discrete decisions and all RNG states are compared exactly, with tight numeric tolerances (`1e-10` for world values, about `1e-6` for float32 network outputs).
- The equivalence tests require complete Git history. If the clone is shallow, run `git fetch --unshallow` first.

## Repository layout

```
main.py, simulation.py        CLI and main simulation engine
*_manager.py                  Parliament, diversity, population, stagnation, aeonic cycle
entities.py, models.py        Entities and PyTorch networks
native_*.py, _tiemu_ffi.py    Native adapters and the C ABI bridge
cpp/                          Standalone C++17 core (CMake)
tests/                        pytest suite, C ABI consumer and frozen fixtures
benchmarks/                   Timing, profiling and comparison tools
docs/                         Performance records and design documents
readme/                       Chinese guide, parameter guide and logging docs
```

## Documentation

- [Release notes](RELEASE_NOTES.md)
- [Performance and equivalence records](docs/performance.md)
- [Design documents and migration log](docs/design/)
- [World parameter guide (Chinese)](readme/Guide.md)
- [Logging guide](readme/clog_en.md)

## Acknowledgements and license

Original project and contributions: YezQiu, HeyCrab3, AvalonRuFae, MoonofBridge24, Akanyi. Logging uses [pyclog](https://github.com/Akanyi/pyclog).

Released under the MIT License; see [LICENSE](LICENSE). This is an unofficial fan project; game names and settings belong to their respective owners.
