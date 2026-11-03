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

- [x] Frozen macOS diagnosis harness/workflow and capacity evidence archived. All three attempts failed to acquire a hosted ARM64 runner before script execution; no measurements exist. No local fallback or further retries.
- [ ] Frozen A2/A3 macOS root-cause validation: uncompleted because the diagnostic could not acquire a cloud runner; do not claim a measured cause or completed acceptance.
- [x] Scoring contracts: `tests/test_resident_scoring_core.py`, `tests/test_resident_scoring.py`; math probe before pytest. Observed cloud RED against A3 and recorded the expected failures.
- [x] C++ scoring: `cpp/tiemu_world.h/.cpp`, C consumer. Source/View and `scoring_prepare(handle,sources,count,matrix,144,view)` / `scoring_finish(handle,token,evaluator,multipliers,count,distribution,distcount,view)`. Reuses core map/score and worker pool; transactional inputs/tokens/locks checked in cloud.
- [x] Python scoring: `native_world.py`, `native_scoring.py`, `native_evaluation.py`, `simulation.py`, evaluator close state. Safe resident path and unchanged fallback; frozen A3 differential/ownership/error-prefix contracts pass on Linux/Windows.
- [x] Scoring measurement: paired baseline frozen at A3; actual C scoring entry points observed without replacing protected evaluator objects. True path entry, integrity and checked digest gates pass.
- [x] Golden lifecycle implementation: metadata-plan C API, original clear/shuffle/write order, cloud RED then strict C/Python state/RNG contracts. Default remains reference selection based on measured full-adapter costs; forced native contracts and paired ablation preserve access to the verified implementation.
- [x] Aeonic index implementation: ordered-pool C ABI and adapter, per-selection and whole-generation contracts, callback refresh/fallback/close. Default gates exclude small populations and live reincarnators based on observed costs; no training algorithm changes.
- [x] Linux/Windows full suites, independent C and CLI, Linux CPU Docker, all three-platform paired checks and Linux profile, independent review and evidence/report completed. Additional speed avenues are discussion only; none implemented.
- [ ] macOS complete suite and scoring microkernel probe: neither attempt acquired a runner; both cancelled before execution. This acceptance remains uncompleted and is not replaced by successful macOS paired performance.
- [x] Final documentation and evidence prepared for a documentation-only `[skip ci]` delivery. No production changes after validated `06353b7`; verify Git delivery and the clean tree after committing/pushing.

## Execution ledger

- Initial checkout clean at `f2073f0`, A branch reused. Memory quick search had no Tiemu entry. Existing A3 report and code supply the current baseline.
- Ruling: user's explicit continuation selects the existing A architecture and ordered priorities. Design details above are implementation decisions, not new user requirements; no redundant architecture approval. Additional broad optimizations are discussion only.
- Independent domains: C++ scoring ABI/contracts; frozen macOS diagnosis; lifecycle/index design. Parent owns Python scoring, workflow/profile integration, cloud execution and evidence. Shared World C++ edits remain serialized.
- Ruling: full native replacement of dot/tanh is not assumed bitwise equivalent. First probe NumPy-compatible batch primitives; keep scalar microkernels where exactness is not demonstrated. No private platform BLAS entry points or reduced precision.

- Scoring RED: `9974dae`, Native run `41650233278`, Linux job `125966122876`: 433 existing passes; missing scoring exports detected. Windows likewise retained 421 passes/12 skips. The new test fixture's rolled caches were changed to owned copies to satisfy the unchanged World input contract.
- Scoring implementation `b98a886`: Linux job `125970870120` passed all 474 tests and C/CLI checks; profile `41651586683` proved real resident prepare/finish entry and unchanged checked digests. Scalar tanh retained; Linux/Windows vecdot probes showed no bit mismatches. Nonfinite/extreme public inputs conservatively use the unchanged A3 path to preserve warning-as-error prefixes.
- Paired workflow initially failed before simulation because checkout treats a short SHA as a branch/tag pattern. `d295393` replaced it with full frozen A3 SHA; paired run `41652508412` then completed Linux/Windows with exact checked digests. Scoring-only single-thread human speed ratios were Linux 1.0247–1.0654, Windows 1.0293–1.0807; these are bounded scenarios, not general or local M1 claims.
- Golden RED: `d295393`, Linux job `125974094883`: 474 existing passes, 30 missing-helper failures and 34 missing-C-export errors. Implemented separately in `b63151f`, including eager identity capture and metadata-only stable top-k.
- Aeonic C RED: `6382f17`, Linux job `125976298939`: all 7 C contracts failed on absent exports. Python RED: `d97eddb`, Linux job `125979343093`: 18 adapter cases failed on missing module, with the 474 preceding cases still passing. `a8f09d2` implements the independent generation-local C index and callback-aware Python scheduling.
- Combined `a8f09d2`: Linux job `125984388814` passed 567 tests, pure C consumer, both CLI thread modes; Docker job `125984388199` passed. Boundary profile `41655444362` passed all integrity/digest gates and observed actual scoring, golden plans, index selects and 15 reincarnator refreshes. These were intermediate checks before the default gates below.
- Frozen macOS attempt 1 (`41650233462`) never acquired a runner. GitHub reported ARM64 capacity constraints, not a simulation failure. A single bounded rerun was authorized within the existing cloud-test scope; no runner architecture or local-compute boundary was changed.
- Attempt 2 also failed before any execution. At 15:06 UTC the final paired macOS job demonstrably acquired ARM64 hardware and completed setup; that external-state change justified one additional attempt of the same frozen diagnostic. No local fallback or architecture switch was used.
- Frozen diagnostic attempt 3 (`126002433112`, 15:07:34–15:22:38 UTC) also never acquired a runner. Exact annotations and all attempts are archived; measurements remain null, no root-cause claim, and retries stopped.
- Intermediate `2bc09b6` passed Linux 574 / Windows 562+12 skips and all C/Docker/CLI checks. Population-only gating was insufficient: Linux 1,000×2 aeonic regressed to 0.859x despite 1,000×12 gaining 1.306x. Full callback rebuilds dominated the short active-training case. `06353b7` defaults to reference selection with a live reincarnator, enables larger non-training candidate pools, and retains forced callback contracts.
- Final `06353b7` Linux job `125998558107`: 579 passed; Windows `125998558823`: 567 passed/12 existing platform skips. Docker `125998558639`, independent C and both CLI modes pass. Profile `41659474030` verifies actual policy decisions: 200-entity cases use reference selection; 1,000×12 uses reference for its active-reincarnator generation and native pools afterward, with identical checked state/RNG. Golden selection is disabled by default and included as an explicit same-candidate performance ablation, not silently deleted.
- Final paired run `41659474466` completed on Linux/Windows/macOS ARM64; every checked digest agrees across all configurations. Default 1,000×12 aeonic speed ratios are 1.438/1.610/1.506. macOS short aeonic and unmodified inorganic adverse results are retained, not attributed to a demonstrated cause. Independent code/claims review completed.
- Final native macOS attempt 1 (`125998557620`) and attempt 2 (`126009410662`, queued from 15:18:57 UTC) both cancelled without acquiring a hosted runner or executing steps. No further retries. The overall native run is not successful; Linux/Windows/Docker success and Mac's separate paired success are recorded with explicit scope in `a4-native-06353b7.json`.
