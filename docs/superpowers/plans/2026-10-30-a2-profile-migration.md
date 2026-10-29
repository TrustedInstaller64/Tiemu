# A2 Cloud Profiling and Ordered C++ Migration Implementation Plan

> **For agentic workers:** Use coordinated implementation and independent review. All execution-based verification is remote on GitHub hosted runners.

**Goal:** Locate A's remaining single-world CPU bottlenecks, migrate the measured safe boundary to C++, and explain the limits of clock-speed and multicore gains.

**Architecture:** Keep the A world/RNG/optimizer ordering contract. Measure uninstrumented and profiled copies of the same deterministic stage, assert matching checked state, then select the smallest substantial state/operation boundary supported by evidence. Following the user's latest instruction, move numerical kernels/state containers into a standalone C++17 core with a C ABI, remove pybind11 from the production build, and move complete inorganic/encounter segments across that interface. Python retains model backends and phase orchestration. Independent rows may run concurrently; dependent interactions remain ordered.

**Tech Stack:** Python 3.12, NumPy 2.2.2, PyTorch 2.12.0, standalone C++17/C ABI, ctypes, GitHub hosted Actions.

**Spec:** `docs/superpowers/specs/2026-10-30-single-world-design.md`, extended by the user's explicit request to continue A with cloud-first profiling. C is deferred and remains unchanged.

## Constraints and acceptance

- No local builds, tests, benchmarks, training, model loading or Docker installation. Local source edits, static checks, small evidence-file IO and Git are permitted.
- No multi-world product runner or batch experiment feature. Repeated deterministic samples exist only to validate performance and equivalence.
- Baseline is A commit `4a2311d`; preserve float64 world arithmetic, float32 networks, RNG consumption, old-cache and ndarray-alias behavior, and every dependent interaction/optimizer step.
- Do not label cProfile inclusive times as additive fractions or use instrumented timings as speedup evidence. Do not infer frequency, IPC or memory boundedness without corresponding measurements.
- Hardware counters are optional diagnostics; preserve permission/unavailability evidence. Do not change host security settings to obtain them.
- Same-platform frozen-upstream replay and real training regression remain required; limited replay does not prove arbitrary long or cross-hardware bitwise identity.

## Ranked hypotheses before measurement

1. Remaining Python encounter dispatch, properties, dirty sets and repeated small pybind/NumPy calls dominate human-stage CPU time. Prediction: profiled self/cumulative hot functions concentrate in these paths, and moving a measured boundary improves unprofiled stage time without changing state.
2. Repeated population scans and transient allocations dominate larger populations. Prediction: scan/property call counts grow with population and remain outside the small native worker region.
3. Real online training is dominated by small PyTorch native/optimizer calls with serial dependencies. Prediction: training profile differs materially from human-stage profile; the human scoring pool cannot explain its time.
4. Native scoring is too small a whole-stage share to scale with four workers, and data preparation/synchronization consumes the savings. Prediction: plain 1/4-worker phase timings barely change even if the native row work is parallel.
5. Cache/branch/memory stalls or lower effective frequency further limit the single active core. Prediction requires phase-scoped PMU/effective-frequency evidence; unavailable counters leave this unresolved.

## Tasks

- [x] Add opt-in phase-only profiling, plain/profiled digest comparison, CPU/environment metadata, and bounded Linux cloud workflow. Keep the existing timing CLI compatible.
- [x] Run the unchanged A runtime remotely; record hotspots, invocation counts, profiling overhead and counter availability. Select migration scope from this evidence.
- [x] Write differential tests for that scope and observe their intended failures remotely before implementing the production API. Test array aliases, ordering, nonfinite/shape inputs, and RNG according to the selected seam.
- [ ] Implement the ordered C++ boundary and Python integration/fallback. Avoid changing stochastic decisions or numerical reduction order.
- [ ] Run Linux/Windows/macOS full suite and Docker remotely, plus same-runner baseline/candidate timing and checked-state comparison. Explore worker counts only for independent work.
- [ ] Independent static review, fix material issues, record evidence and performance limits, commit and push A. Keep C untouched.

## Execution record

- User explicitly authorized this continuation and cloud verification. The existing A design is reused; no redundant design approval is required. Local prohibition overrides skill instructions that would otherwise run tests locally.
- Initial A tree clean at `4a2311d`; no project-specific AGENTS.md/GLOSSARY.md/ADRs found. Static hotspot analysis, official CPU documentation research and profiling harness work are delegated independently.
- Instrumentation is a bounded diagnostic tool verified by its own plain/profiled state comparison; production migration will retain remote RED/GREEN gates.
- Initial profile at `39f2841` passed observer/state checks but exposed pybind11/cProfile frame loss (upstream issue #5600): missing Python calls and self time greater than cumulative time. Those percentages are rejected. Independent explicit boundary timers at `6badfd7`, run `40960718586`, passed call-count integrity and all four plain/boundary/worker state comparisons. PMU access was unavailable (`perf_event_paranoid=4`); frequency/IPC/cache causality remains unmeasured.
- Measured scope: human 10,000 x 3 had 15,000 opponent selections, 14,120 applied interactions and 216,333 dominant-path queries. The native scoring call occupied milliseconds while the plain whole stage was about one second. Prioritize the complete encounter loop and cache state over adding workers to that small scoring region.
- User explicitly changed the architecture to important computation modules fully in C++ without pybind11. Production core and headers will have no Python/NumPy/binding API dependency; a C-only smoke client proves standalone execution. Frozen A1 builds may still install pybind11 solely to provide the historical remote oracle.
- C ABI/inorganic engine RED at `6badfd7`: Linux 245 passed, 18 intended xfails restricted to missing attributes; all platform/Docker jobs succeeded (`40960718719`). Core arithmetic/state handles, MT state import/export and whole inorganic grid segment implementation started only after this result.
- Encounter RED contracts at `18d1466` compare frozen A1's per-attempt selections/culls, full Python/NumPy MT states, cache/array identities, logs and unsafe-input fallback. Remote `40962458919` completed on all platforms and Docker; Linux had 245 passed and 43 intended xfails (18 core + 25 encounter). Encounter implementation began after that result.
- Static review found stale `_tiemu_native*.so` files could shadow the adapter. Production imports the new `_tiemu_ffi` name, with an upgrade regression test; old binary files cannot silently select the pybind backend.
- Foundation GREEN at `534c984`, run `40963299160`: Linux/macOS 264 passed with the 25 encounter contracts still intentionally pending, all platform/Docker jobs successful and standalone CMake/C consumer successful. The source no longer depends on pybind11. Complete inorganic stage comparisons preserve legacy/RNG on all three platforms and improve 7.68–11.46x. Human loop still used fine-grained ctypes at this intermediate commit and regressed; it is not an accepted final performance outcome.
- Independent encounter review identified default newborn integer-zero scores disabling the whole-stage fast path, scalar-type changes on zero-damage battles, and the exceptional Python division case at score -1e-6. Cover these before final acceptance; the boundary profile must prove each normal human generation actually entered the native engine.
