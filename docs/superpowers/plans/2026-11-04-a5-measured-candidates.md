# A5 Measured Candidate Selection Plan

> **For agentic workers:** Coordinated implementation and independent review. Every project execution is cloud-only; the user's local-load prohibition overrides general local test commands.

**Goal:** Select faster A implementations from measured, equivalent candidates; distinguish continuous training from the existing full-cycle workload.

**Architecture:** Compare the same A4 checkout with its current or frozen A3 generation loop in isolated processes. Separately compile a macro-only inorganic neighbor-table candidate against frozen A4. Keep full outputs/RNG/training state checks, balanced order and a same-code noise control. Grok ideas are inputs for review, not an equivalent baseline.

**Tech Stack:** C++17/C ABI, Python 3.12, NumPy 2.2.2, Torch 2.12.0 CPU, hosted Linux/Windows/macOS Actions.

**Spec:** User's November 4 requests to assess regressions, inspect the Grok submission, benchmark alternatives and choose speed; existing A equivalence contract in `docs/superpowers/specs/2026-11-03-a4-scoring-design.md`.

## Constraints and decisions

- Reuse `codex/single-world-a`; freeze A4 at `e8b73873e9938fd068196d54e26881cc36539750`, legacy generation method at `f2073f001af4bc3f03da6d96d5360aa9a2027e0a`.
- No local project imports, builds, tests, simulation, training or timing; no local Docker. Static edits/Git/small evidence parsing only.
- Preserve precision, sequential updates, population order, full RNG state, optimizer steps and observable mutation semantics. Do not silently fix the inherited reincarnator-reference defect: doing so changes work/results.
- Full-cycle 12 generations is not 12 generations of sustained training. Record actual per-generation live reference, population and optimizer increments outside pure timing. Controlled sustained training uses a transparent common fixture and is reported separately.
- Neighbor candidate defaults off until evidence supports admission. Include table construction and small-call regressions; byte-compare complete cells and MT continuation.
- Performance-only refactor exception: valid differential tests should pass on the baseline too. Write tests first and add intentional wrong-algorithm negative controls; do not invent a missing-feature RED for an equivalent implementation.
- Existing authorization covers cloud comparison and commit/push. New substantial state-management rewrites and fixes changing original behavior are not included.
- Found Grok in the untracked evaluation submission of worktree `9179`, not a published branch. It is based on `5299d0c`, while A retains `691c6e0` rules. Do not rank their human-stage timings as equal-work until the rules are aligned. Leave the external submission untouched.

## Review focus

1. A candidate must not become faster by ceasing actual training or changing the simulated workload.
2. Method replacement must affect only the intended generation loop; protected selector/interaction identities stay unchanged.
3. Full raw state, array order, model and optimizer contents, and all RNGs must match before a timing result is accepted.
4. Neighbor accumulation order and in-place visibility must match, including grid 1, zero steps and chunk restarts.
5. Every child must load its own checkout/library; compiler logs must show the candidate macro, especially on Windows. Same-code controls must constrain claims from noisy short samples.

## Tasks

- [x] Read-only A4 and Grok source/evidence review. Existing small-population/live-reincarnator selector fallback confirmed; old reference defect identified separately.
- [x] Add `tests/test_inorganic_neighbors.py` with frozen algorithm and wrong-order/synchronous-update negative controls; implement opt-in `TIEMU_PRECOMPUTE_NEIGHBORS` in `cpp/tiemu_inorganic.cpp`. Default 34 contracts and macro full suites passed on all three hosted platforms; independent review completed.
- [x] Add `benchmarks/compare_aeonic_structures.py`: current/legacy/current-repeat; full-cycle and controlled-training cases; six balanced blocks, two fresh samples, separate observers and strict consistency gates. All 24 platform/case combinations agree; 864 raw timing samples archived.
- [x] Add `benchmarks/compare_inorganic_neighbors.py`: baseline/candidate/baseline-repeat; bounded kernel cases and actual headless 100-step chunk phase, complete output/MT and legacy/RNG checks. All three platforms agree; 1,944 raw samples archived.
- [x] Add `.github/workflows/a5-candidate-performance.yml`: frozen/default and macro builds, compile-flag verification, relevant full tests and same-runner comparisons on all three platforms. Run `41794212195` succeeded on all platforms. Default Native run `41794212208` and legacy paired run `41794212221` also succeeded.
- [x] Inspect cloud results and independent review. Retain A4 default: no stable continuous-training benefit from the old loop; neighbor table has no cross-platform actual-phase benefit and regresses single-step calls. The opt-in experiment remains disabled.
- [x] Record current bottlenecks, precise short/long-training limits, source-backed Grok assessment and per-platform evidence. Final docs/evidence prepared for commit/push; verify clean Git delivery after that documentation-only commit.

Additional bounded diagnostic: with new actual Mac runner availability, dispatch `41794495133` successfully reran the frozen A2/A3 experiment. Prior short regression did not stably reproduce; same-code variation is large. Zero GC collections and zero World birth calls only bound this fixture/window, not the universal cause. No retry loop or local fallback.

## Falsifiable comparisons

- If current Python generation-wrapper overhead causes small-world regression, replacing only that method with the frozen original should improve those same workloads beyond current/current-repeat variability.
- If native candidate indexing helps only larger non-training populations, the full-cycle large case should gain while controlled live-reincarnator training shows little/no index benefit; observe actual entries separately.
- If repeated neighbor geometry is material at real call sizes, table construction included in 100-step calls and the complete inorganic phase should still net a repeatable gain. One-step regressions count against unconditional adoption.
- If short macOS differences are dominated by run variation, identical-code controls will show comparable shifts; that supports uncertainty, not a specific thermal/GC/scheduling cause.
