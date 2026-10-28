# Tiemu native compute and platform design

User authorization: modify Tiemu directly and submit to git@github.com:TrustedInstaller64/Tiemu.git. AMPHOREUS is retained untouched. Baseline: 691c6e0023a83513ad3cd386cfee2198317104d3.

## Goals and evidence

Replace measured Python numerical/collection hotspots with C++17 exposed via pybind11. Keep the Python application and existing native NumPy/PyTorch/LLM libraries. In the original human-phase 1000-entity, two-generation sample, grouped timing measured 73.83% global-path-distribution work and 19.15% opponent selection; this does not prove 95% pure interpreter overhead. Reproduce this workload before and after and report real timing, not an expected speedup.

## Approach

Use a native extension plus the original Python reference backend. A complete standalone C++ application would duplicate model integration and control logic; only replacing tiny matrix multiplies would miss the measured bottlenecks. The native extension handles bounded numerical kernels, ordered candidate indexing and incremental path histograms. Python handles random draws, sequencing, model calls, I/O and control. No fast-math or unordered parallel reductions. Releases of the GIL are limited to code that does not access Python objects.

## Semantic contract

- Preserve original encounter counts, ordered candidate pools (including original stale-bin behavior), per-encounter purification and normalization, identity vote weights, and sequential in-place inorganic updates.
- Keep Python random and NumPy RNG draw order; add explicit seed configuration. C++ consumes supplied noise/random ranks. Do not change optimizer, loss, precision or batch size.
- Reference/native kernels use float64 for world arithmetic, float32 for neural networks. Compare arithmetic with tight tolerances and discrete decisions with exact equality on fixed fixtures. Cross-device training and independently sampled LLM outputs are not promised bit-identical.
- Cache path histograms only within the interaction loop. Initialize lazily at the original first distribution query, refresh dirty participants before the next query, and discard the cache before global interventions/population changes. No stale cache after a population mutation.
- Keep the Python backend available for differential tests. `auto` may fall back with a clear warning; explicitly requested `cpp` fails when the extension is missing.

## Runtime platforms

- Apple Silicon macOS: local MLX model for LLM via Metal; PyTorch MPS when available. Explicit CPU selection remains possible.
- Windows native: GGUF llama.cpp with its own CUDA-capability check; PyTorch CUDA when available, CPU otherwise. No mandatory MLX installation.
- Hyper-V Linux VM/Docker: CPU native extension, CPU PyTorch and GGUF CPU support; default image can run without an LLM/model. No assumption of host GPU passthrough.
- Separate `torch_device`, `llm_backend`, model paths and LLM GPU layer configuration. Explicit unavailable choices fail clearly; auto decisions are logged. No implicit model downloads.

## Console and lifecycle

One owner reads stdin. Windows uses msvcrt; POSIX uses cbreak/select with TTY guarding and terminal restoration. Paused command input must not race a hotkey reader. Non-TTY operation is supported without opening keyboard devices or requiring root. `--no-interactive`, stop command, SIGINT/SIGTERM and Windows SIGBREAK reach bounded safe points in all phases. C++ calls are at most one generation/chunk. Exit with truthful return codes; no unconditional Windows pause.

Stop exports a JSON world snapshot and CPU-portable policy weights to the configured output directory. Existing JSON is not a complete training checkpoint: phase internals, optimizers and RNG are not all restored. Document this limit; do not advertise exact resume. Atomic replacement prevents partially written snapshots.

## Packaging and validation

Python 3.11/3.12, C++17, setuptools/pybind11 build; optional MLX and llama.cpp dependencies. Linux CPU multi-stage Docker image, mounted models/output, no embedded model files. Windows/MSVC, macOS/Clang, Linux/GCC CI builds run actual native tests. Unit differential tests, frozen original replay, PTY and non-TTY tests, backend capability/selection tests, short CLI runs, benchmark and fresh review precede submission.

Local verification limits must be explicit: Docker CLI exists but daemon is unavailable; Windows/CUDA and MPS require their real hosts. CI results may extend validation but a mocked API is not real GPU acceptance.
