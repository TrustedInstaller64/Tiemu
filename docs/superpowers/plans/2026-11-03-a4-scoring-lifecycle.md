# A4 Scoring and Lifecycle Implementation Plan

> **For agentic workers:** Use coordinated execution with independent review. All executable verification is remote; the user's local-compute prohibition overrides local test commands in general skills.

**Goal:** Move scoring preparation/cache work, golden candidate planning and aeonic candidate indexing into native stages while preserving A results; diagnose the macOS short aeonic regression.

**Architecture:** Reuse the resident C++ World with independent scoring and lifecycle workspaces. Retain original numerical microkernels and ordered Python RNG where necessary. Extend only the explicitly selected A migration; other optimization ideas remain discussion.

**Tech Stack:** C++17/C ABI, ctypes, Python 3.12, NumPy 2.2.2, PyTorch 2.12.0, GitHub hosted Actions.

**Spec:** `docs/superpowers/specs/2026-11-03-a4-scoring-design.md`.

## Global constraints

- A only; baseline `f2073f0`/`5714a60`; C/main/AMPHOREUS untouched.
- No local project execution/import/build/test/profile, model loading or Docker installation.
- Preserve float64/float32, RNG, optimizer order, public mutations, aliases, cache timing and failure prefixes.
- Production remains independent C++17/C ABI; no pybind11, private BLAS ABI, or product batch-experiment scheduler.
- Existing authorization covers execution/commit/push of the selected A steps; no repeated permission gate for routine implementation choices.

## Review focus

1. Old cached paths must supply multipliers before repair/current values generate new paths.
2. Cached seat values and source arrays can be directly modified; identity is not a dirty detector.
3. Nonfinite rows, custom objects, errors and closed handles must preserve ordered prefix behavior.
4. Borrowed arrays, tokens and stable IDs must survive synchronous calls without exposing stale numeric state.
5. Stable ties, duplicate reincarnator weight, culled draws and callback mutations must retain exact RNG/selection behavior.

## Tasks

- [ ] Frozen macOS diagnosis: new `benchmarks/diagnose_macos_aeonic.py` and workflow. Validate actual training and identical-copy variation; archive evidence without production fixes until a cause is supported.
- [ ] Scoring contracts: `tests/test_resident_scoring_core.py`, `tests/test_resident_scoring.py`; math probe before pytest. Observe cloud RED against A3 and record the expected failures.
- [ ] C++ scoring: `cpp/tiemu_world.h/.cpp`, C consumer. Add Source/View and `scoring_prepare(handle,sources,count,matrix,144,view)` / `scoring_finish(handle,token,evaluator,multipliers,count,distribution,distcount,view)`. Reuse core map/score and worker pool; validate transactional inputs/tokens/locks.
- [ ] Python scoring: `native_world.py`, new `native_scoring.py`, `native_evaluation.py`, `simulation.py`, evaluator close state if needed. Add `try_evaluate_resident(entities,parliament,distribution,evaluator,world)->bool`; only safe cases enter it. Run frozen A3 differential/ownership/error-prefix contracts and full cloud suite.
- [ ] Scoring measurement: advance paired baseline to frozen A3; instrument actual C scoring entry points without replacing protected evaluator objects. Confirm the measured world uses the new stage and compare same-runner results.
- [ ] Golden lifecycle: `native_lifecycle.py`, `population_manager.py`, World metadata-plan C API and tests. Native filter/stable-sort prefix, original clear/shuffle/write order. First cloud RED, then implementation and strict state/RNG comparisons.
- [ ] Aeonic index: finalize the ordered-pool C ABI and adapter, first per-selection/whole-generation contracts, then implementation. Refresh all metadata after actual reincarnator interaction; update pair after ordinary interaction; preserve fallback and close on exceptions. No training algorithm changes.
- [ ] Complete three-platform/C/Docker/paired/profile checks, independent review, evidence/report, commit/push with clean tree. Discuss additional speed avenues separately; implement none of those suggestions.

## Execution ledger

- Initial checkout clean at `f2073f0`, A branch reused. Memory quick search had no Tiemu entry. Existing A3 report and code supply the current baseline.
- Ruling: user's explicit continuation selects the existing A architecture and ordered priorities. Design details above are implementation decisions, not new user requirements; no redundant architecture approval. Additional broad optimizations are discussion only.
- Independent domains: C++ scoring ABI/contracts; frozen macOS diagnosis; lifecycle/index design. Parent owns Python scoring, workflow/profile integration, cloud execution and evidence. Shared World C++ edits remain serialized.
- Ruling: full native replacement of dot/tanh is not assumed bitwise equivalent. First probe NumPy-compatible batch primitives; keep scalar microkernels where exactness is not demonstrated. No private platform BLAS entry points or reduced precision.
