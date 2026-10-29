# Single-world A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate one world while preserving ordered evolution, on isolated A and C branches.

**Architecture:** Fuse pure native numerical pipelines and batch only independent scoring rows. Keep Python RNG and dependent interactions ordered, expose a bounded native worker budget, and perform all execution-based verification remotely.

**Tech Stack:** Python 3.12, NumPy 2.2.2, PyTorch 2.12.0, C++17/pybind11, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-30-single-world-design.md`

## Global Constraints

- No local compilation, tests, benchmarks, training, model loading or Docker installation.
- No independent-world parallelism or batch-experiment scheduler.
- A uses ordered world float64/network float32 arithmetic and original RNG/optimizer sequencing.
- C remains a separate long-term branch, with an explicit divergence ledger.

## Review Focus

- Direct mutation/replacement of parliament seats and total_seats must invalidate cached vectors (Task 2).
- Nonpositive purification and external ndarray aliases must retain old side effects (Tasks 1/2).
- Cached old paths must remain the source of multipliers before scoring refresh (Task 2).
- Thread pool close, concurrent callers, empty rows and malformed shapes must not deadlock or access invalid memory (Task 1).
- Inference-only autograd removal must not change actions, RNG, Adam states or trained parameters (Task 3).

### Task 1: Fused and parallel native primitives

**Files:** `cpp/tiemu_native.cpp`, optional private C++ header, `tests/test_native_pipeline.py`.

**Interfaces:** `normalize_pipeline(values, factor, tide, max_norm) -> ndarray`; `BatchEvaluator(workers=1,min_rows=4096).concepts(values,matrix,multipliers,distribution,bosses) -> (paths,stats)`; `close()` idempotent. stats columns are activity, stability, purity, score.

- [x] Add differential tests against existing scalar native primitives, input immutability, nonfinite/boss/shape/empty/close cases and bitwise 1/2/4-worker comparisons.
- [x] Verify remotely that tests fail because the new APIs are absent.
- [x] Implement ordered fused kernels and lazy C++ worker pool; reuse vote templates without reordering accumulation.
- [x] Remote full suite and native lifecycle review; commit the passing change.

### Task 2: Integrate one-world batch scoring

**Files:** `native_evaluation.py`, `population_manager.py`, `parliament_manager.py`, `simulation.py`, `main.py`, `config.py`, `tests/test_native_evaluation.py`.

**Interfaces:** `evaluate_entities(entities, parliament, distribution, evaluator=None)`; `AeonEvolution.close()`; CLI `--native-threads` and `--native-min-rows` (positive integers, defaults 1/4096).

- [x] Add tests for old-path multipliers, scalar/batch equality, invalid titan values, seat dictionary mutations and preserved aliases.
- [x] Verify missing evaluator integration remotely before implementation.
- [x] Integrate fusion only on alias-safe branches; batch fixed-distribution scoring and close the lazy evaluator on exit.
- [x] Keep original scalar fallback and full same-platform original replay; commit passing change.

### Task 3: Ordered training overhead

**Files:** `aeonic_cycle_manager.py`, `tests/test_training_overhead.py`.

- [x] Add real-network tests showing inference forward retains an unnecessary graph, plus seeded state/action/optimizer/RNG differential checks.
- [x] Verify the intended graph-overhead failure remotely.
- [x] Reuse one state tensor, disable gradients only for sampling, remove redundant retain_graph; preserve advantage Python-float computation and step order.
- [x] Run full phase replay remotely and commit.

### Task 4: Runtime controls and remote evidence

**Files:** `main.py`, `config.py`, `.github/workflows/native.yml`, remote performance workflow, bounded benchmark scripts, `docs/performance.md`, Chinese runtime documentation.

- [x] Add native-worker CLI validation and a compute log mode that preserves message argument side effects.
- [x] Add remote paired-baseline measurement, including actual training, with machine/scope metadata and downloadable results; no local execution.
- [x] Run Windows/Linux/macOS full regression and Linux Docker checks remotely.
- [x] Fresh independent diff review; repair material issues; publish A and a separate C roadmap branch, no merge to main.

## Execution record

- 2026-10-30: User approved A first, C separate/long-term, C++ migration exploration and single-world multicore. No further design confirmation required; implementation uses coordinated agents and a final independent review. Local restriction overrides local test instructions: RED/GREEN are remote only.
- Baseline: `1ce39e00b8808564a331861ffcc799d4237ac324`. Existing checkout reused; initial tree clean. Git reports an unrelated malformed ref name `codex/native-cross-platform 2`; left untouched.
- Branch isolation: A `codex/single-world-a`; C `codex/experimental-c`. Common design at `2f06b3a`; C roadmap at `240b277`, without experimental runtime changes. Neither branch merged into main.
- Remote RED at `f942b5a`: run [40952329166](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40952329166), existing 131 passed and new contracts 93 intended failures/18 passed. Missing APIs/integration/gradient assertions were checked explicitly.
- Tasks 1–3 and runtime integration landed together at `f9aafcb`. First remote GREEN: Linux/macOS 244 passed; Windows 232 passed/12 POSIX skipped; CPU Docker passed. No local test or compiler invocation.
- Independent review identified hidden paused-console help/status in compute-only. Fixed in `dacb888` with real command-parser coverage; native lifecycle/locking review found no blocking defect. Runtime documentation followed in `0af00fc`.
- Final executable verification at `22b1620`: [regression run 40954826520](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40954826520), Linux/macOS 245 passed, Windows 233 passed/12 skipped and CPU Docker passed. [Performance run 40954826586](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40954826586) completed on all three hosted platforms, with identical checked state/RNG digests for baseline/A1/A4/A4-forced.
- Results are recorded in `docs/performance.md` and `docs/performance/single-world-2026-10-30-*.json`: complete Windows/macOS artifacts with verified ZIP hashes, and Linux summaries from successful job logs. The Linux connector download was not a valid ZIP after retry; its invalid payload was discarded, and raw Linux repetitions remain in Actions. First pool startup is included in timed samples; aeonic does not exercise the human batch-scoring pool. Default native budget stays 1 because four workers did not consistently improve whole-stage performance. Cloud macOS variability is not presented as this M1's measured speed.
- This plan's first A increment is complete. Whole-engine C++ migration, NUMA scheduling, GPU world kernels and C's numerical experiments remain future stages with their own equivalence gates; this record does not claim their completion.
