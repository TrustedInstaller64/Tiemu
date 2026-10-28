# Tiemu Native Compute and Platform Implementation Plan

> **For agentic workers:** Use focused parallel implementation with explicit file ownership; root integrates and a separate reviewer checks the final diff. Follow test-first changes and preserve the reference implementation.

**Goal:** Accelerate original Tiemu without changing its world rules, and support CPU Linux Docker plus native Windows/macOS model and console backends.
**Architecture:** C++17/pybind11 numerical kernels and ordered indices; Python controls sequencing/RNG. Independent model/device selection and terminal lifecycle.
**Tech Stack:** Python 3.11/3.12, NumPy, PyTorch, pybind11, optional mlx-lm/llama-cpp-python, pytest, Docker.
**Spec:** docs/superpowers/specs/2026-10-28-native-platform-design.md

## Global constraints

Preserve original counts, random order, sequential updates, vote weights and float32 training. No fast-math, platform-specific CPU ISA, implicit downloads or required GPU. AMPHOREUS untouched. Explicit unsupported backends fail; automatic fallback is reported.

## Review focus

- Incremental histogram invalidation when interactions change paths and when population changes.
- Original candidate ordering and RNG state, including dead candidates retained in original bins.
- Non-TTY input, paused input ownership, terminal restoration and stop in each phase.
- Missing LLM packages/models and independent llama.cpp/PyTorch GPU capabilities.
- Fresh wheel installation and CPU Docker start without repository-relative output assumptions.

## Tasks

- [ ] Baseline: archive fixed original source outside worktree; record profile and add build metadata before implementation commits.
- [ ] Native kernels: tests first for mapping, concepts, purification, norm cap, weighted votes, sequential inorganic step, ordered candidates and incremental histogram. Add cpp/tiemu_native.cpp and native_backend.py; root owns packaging. Build extension and run native differential tests.
- [ ] Console: tests first for POSIX PTY, pipe/non-TTY, paused command handling and cleanup. Own console_control.py/debugger.py/tests/test_console.py. Root wires safe points and signals in simulation/main.
- [ ] Model backends: tests first for auto/explicit device choices, local model checks, disabled lazy imports, MLX sampler/chat template and GGUF GPU capabilities. Own device_utils.py/cpu_llm_interface.py/tests/test_backends.py. Root wires device tensors and config.
- [ ] Integration: native/reference selection in entities/models/population/parliament/interaction/simulation, seeded replay against archived original, bounded CLI execution and no-interactive/signal lifecycle. Tests for actual trajectories and RNG states, not only kernel self-comparison.
- [ ] Delivery: Docker CPU and optional GGUF build, platform CI and Chinese run instructions. Benchmark identical workload including crossing/copy costs. Fresh review, full tests, install/build smoke, commits and push to user repository without force.

## Execution notes

2026-10-28: User explicitly requested direct changes and repository submission. Proceed without additional design approval rounds. Root coordinates commits; agents do not commit or edit each other's files.
