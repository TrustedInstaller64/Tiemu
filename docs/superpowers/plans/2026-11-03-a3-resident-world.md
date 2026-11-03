# A3 Resident World Implementation Plan

> **For agentic workers:** Use coordinated implementation with independent review. All execution-based verification is remote; local source edits, static inspection and Git only.

**Goal:** Make a persistent C++ world participate in normal ordered encounters and entity lifecycle, move newborn numeric work as a batch, and diagnose the aeonic regression with measured evidence.

**Architecture:** An opaque C++17 World owns contiguous state and stable IDs. Explicit synchronization observes Python's mutable public arrays before native stages; Python retains independent mirrors and original alias semantics. Birth arithmetic moves across one C ABI call while RNG, selection and optimizer ordering remain intact.

**Tech Stack:** C++17/C ABI, ctypes, Python 3.12, pinned NumPy/PyTorch, GitHub hosted Actions.

**Spec:** `docs/superpowers/specs/2026-11-03-resident-world-design.md` and the existing A design.

## Global constraints

- A only, baseline `eca7377`/`26c1c26`; no main/C/AMPHOREUS changes.
- No local project execution/build/test/profile/model loading/Docker installation.
- Preserve float64/float32, RNG, order, old array aliases, lazy caches and online training updates.
- No pybind11 production dependency; no multi-world product scheduler.
- Keep finite replay and cloud hardware evidence boundaries explicit.

## Review focus

1. Direct Python in-place writes and stale caches must be imported faithfully, not inferred from identity.
2. Removed/reordered/replaced entities and same-name births must not reuse stable IDs or corrupt population order.
3. Newborn integer zeros, lazy caches, RNG Gaussian cache and name collisions must match the reference.
4. Borrowed buffers must survive synchronous C calls; failed validation must not partially commit and errors after mutation must not trigger fallback.
5. Instrumentation must not disable a fast path, affect RNG, or turn tiny noisy samples into causal performance claims.

## Tasks

- [x] Add bounded frozen-A1/A2 aeonic diagnosis and run it remotely. Record rotated pure timings, separate inclusive spans, optimizer-step evidence and checked-state equality for each ablation.
- [ ] Fix World C ABI and Python interfaces between implementers. Add lifecycle/birth behavior contracts first; observe missing implementation failures in cloud before production changes.
- [ ] Implement `cpp/tiemu_world.h/.cpp`, add standalone/build integration and C consumer contracts. World owns state and IDs, validates before sync commit, exposes only bounded synchronous borrowed data.
- [ ] Implement `native_world.py` and actual `native_encounters.py`/`simulation.py` lifetime integration. Test reuse, reorder/delete/add, external mutation, close/load and unchanged original encounter behavior.
- [ ] Implement the batched newborn pipeline in `population_manager.py` using the World API. Test frozen-reference arrays, scalar types, names, cache state and full RNG continuation, plus unsupported-case fallbacks.
- [ ] Apply only measured aeonic improvements with focused differential tests. If the regression cannot be isolated or eliminated, retain that limitation rather than claiming a fix.
- [ ] Update profiling integrity for the resident engine, compare same-runner A2/candidate, run full Linux/Windows/macOS/C/Docker regression in cloud and independently review the final diff.
- [ ] Record absolute timings, speedups/regressions, equivalence boundaries and remaining work. Commit/push A and confirm a clean tree; leave C untouched.

## Execution ledger

- Initial tree clean at `eca7377`, branch `codex/single-world-a`; existing A2 cloud results retained.
- Ruling: the user's explicit continuation authorizes the already selected A architecture and remote verification; no redundant design/commit/push approval. The prohibition on local load overrides generic skill commands that execute tests locally.
- Ruling: keep complete synchronization at Python-visible boundaries because public ndarrays can change in place. Claim persistent native ownership and actual native stage execution, not complete Python removal or zero-copy whole-world management.
- Independent ownership: resident C++ ABI/build/C smoke; Python integration/birth and contracts; aeonic diagnostic harness/workflow. Parent owns performance/profile workflow integration and evidence.
- Remote RED at `0047daa`, run `41634794869`, Ubuntu job `125913148001`: original 346 tests pass; 22 new failures and 19 setup errors all identify the missing `native_world`/World C ABI/lifecycle API. Production implementation began after observing this result. Additional reviewer-requested edge cases cover positional IDs after reordering, invalid native calls, load teardown and birth exception prefixes.
- First frozen-A1/A2 diagnosis at `42b8fdf`, run `41634508599`: all pure/observed/ablation checked states agree. The 200×2 process-median time is 0.015565/0.016004 s; 200×12 is 0.023081/0.028540 s. Both scenarios perform 15 actual updates per optimizer, so the longer case adds selection/ordinary interactions rather than more training. A2's selector span grows relative to A1 while the training span is similar; neither legacy distribution nor disabling GC removes the difference. Further causal ablations are required before naming a root cause.
