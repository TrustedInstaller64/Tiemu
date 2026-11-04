"""Hosted-only paired neighbor-table experiment; this script does not build cores.

The workflow builds two isolated roots and proves the candidate macro in its
compiler log. This driver runs their native kernels and actual inorganic phase
in fresh processes, including a second same-code baseline as a noise control.
"""
import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


CASES = tuple((grid, steps) for grid in (1, 2, 10) for steps in (1, 100, 10000)) + (
    (32, 1), (32, 100))
CONFIGURATIONS = ("baseline", "candidate", "baseline-repeat")
ORDERS = tuple(itertools.permutations(CONFIGURATIONS))
THREAD_ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                  PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def array_state(activity, stability, mt):
    # Hash every byte, never array reprs or just the final mean. Shape and dtype
    # are part of each digest's contract; MT includes all 624 words and index.
    result = {}
    for name, value in (("activity", activity), ("stability", stability), ("mt", mt)):
        result[name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                        "byte_count": value.nbytes,
                        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    result["state_digest"] = digest_json(result)
    return result


def kernel_worker(args):
    os.environ.update(THREAD_ENV)
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    native = importlib.import_module("_tiemu_ffi")
    core = Path(native.__core_file__).resolve()
    if (Path(native.__file__).resolve().parent != root or core.parent != root
            or native.__binding__ != "c-abi" or native.ABI_VERSION != 1):
        raise RuntimeError("Worker did not load the selected root's in-place C ABI core")
    function = native._inorganic_run
    case_order = CASES[args.block % len(CASES):] + CASES[:args.block % len(CASES)]
    results = []
    for grid, steps in case_order:
        cells = grid * grid
        rng = random.Random(731)
        rng.gauss(0.0, 1.0)
        values = [(rng.uniform(1.0, 5.0), rng.uniform(1.0, 5.0)) for _ in range(cells)]
        activity = np.array([row[0] for row in values], dtype=np.float64)
        stability = np.array([row[1] for row in values], dtype=np.float64)
        mt = np.array(rng.getstate()[1], dtype=np.uint32)
        before = array_state(activity, stability, mt)
        out_activity, out_stability, out_mt = np.empty_like(activity), np.empty_like(stability), np.empty_like(mt)
        arguments = (activity.ctypes.data, cells, stability.ctypes.data, cells,
                     mt.ctypes.data, mt.size, steps, grid,
                     out_activity.ctypes.data, cells, out_stability.ctypes.data, cells,
                     out_mt.ctypes.data, out_mt.size)
        calls = min(args.max_calls, max(1, args.target_cell_updates // (cells * steps)))
        native._check(function(*arguments))  # One excluded warm-up per case/process.
        samples = []
        for index in range(args.samples):
            start_cpu, start = time.process_time(), time.perf_counter()
            for _ in range(calls):
                status = function(*arguments)
                if status:
                    native._check(status)
            elapsed = time.perf_counter() - start
            cpu_elapsed = time.process_time() - start_cpu
            if elapsed <= 0 or not math.isfinite(elapsed):
                raise RuntimeError("Kernel timer returned a nonpositive or nonfinite duration")
            if array_state(activity, stability, mt) != before:
                raise AssertionError("Native kernel mutated its input buffers")
            samples.append({"sample": index, "calls": calls, "seconds": elapsed,
                            "cpu_seconds": cpu_elapsed, "seconds_per_call": elapsed / calls,
                            "cell_updates": cells * steps * calls,
                            "checked_state": array_state(out_activity, out_stability, out_mt)})
        results.append({"case": f"grid{grid}-steps{steps}", "grid": grid, "steps": steps,
                        "input_state": before, "samples": samples})
    pointer_bytes = np.dtype(np.uintp).itemsize
    # NeighborRow is eight size_t indices plus uint8 count rounded to size_t
    # alignment. This excludes vector/allocator overhead and is not an RSS probe.
    estimated_row_bytes = ((8 * pointer_bytes + 1 + pointer_bytes - 1) // pointer_bytes) * pointer_bytes
    report = {
        "schema_version": 1, "mode": "kernel", "configuration": args.configuration,
        "block": args.block, "source_root": str(root), "core_file": str(core),
        "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           text=True, encoding="utf-8").strip(),
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "logical_cpus": os.cpu_count(), "thread_environment": THREAD_ENV,
        "candidate_macro_proof": "External compiler log; a different binary digest alone does not prove the macro.",
        "estimated_table_bytes_per_cell": estimated_row_bytes,
        "table_memory_scope": "Layout estimate only, not measured allocation or peak RSS; candidate allocates per call.",
        "timing_scope": "Direct C ABI calls with preallocated buffers and pointers; includes C table construction, "
                        "grid copying, ordered arithmetic, RNG and ctypes/Python call-loop overhead. Excludes imports, "
                        "input/output allocation, warm-up and state hashing. Every call restarts from the same input "
                        "and MT state; tiny cases are call-bound and are not a C-only stopwatch measurement.",
        "cases": results,
    }
    write_json(args.worker_output, report)


def descriptive(values):
    ordered = sorted(values)
    return {"count": len(values), "median": statistics.median(values),
            "min": ordered[0], "max": ordered[-1],
            "q1": statistics.median(ordered[:len(ordered) // 2]),
            "q3": statistics.median(ordered[(len(ordered) + 1) // 2:])}


def summarize_case(records, case, blocks, mode):
    medians = {name: [] for name in CONFIGURATIONS}
    all_samples = {name: [] for name in CONFIGURATIONS}
    digests, input_digests = set(), set()
    per_block = []
    for block in range(blocks):
        row = {"block": block, "order": list(ORDERS[block % len(ORDERS)]), "median_seconds": {}}
        for name in CONFIGURATIONS:
            record = records[(block, name)]
            if mode == "kernel":
                result = next(value for value in record["cases"] if value["case"] == case)
                values = [sample["seconds_per_call"] for sample in result["samples"]]
                states = [sample["checked_state"] for sample in result["samples"]]
                input_digests.add(result["input_state"]["state_digest"])
            else:
                values, states = record["seconds"], record["checked_states"]
            if not values or any(value <= 0 or not math.isfinite(value) for value in values):
                raise ValueError("Every timing sample must be finite and positive")
            all_samples[name].extend(values)
            value = statistics.median(values)
            medians[name].append(value)
            row["median_seconds"][name] = value
            digests.update(state["state_digest"] for state in states)
        timings = row["median_seconds"]
        row["baseline_over_candidate"] = timings["baseline"] / timings["candidate"]
        row["baseline_repeat_over_candidate"] = timings["baseline-repeat"] / timings["candidate"]
        row["baseline_over_baseline_repeat"] = timings["baseline"] / timings["baseline-repeat"]
        per_block.append(row)
    return {"case": case, "timing_unit": "seconds_per_C_ABI_call" if mode == "kernel" else "seconds_per_actual_phase",
            "seconds_by_configuration": {name: descriptive(values) for name, values in all_samples.items()},
            "median_of_block_medians": {name: statistics.median(values) for name, values in medians.items()},
            "paired_speedup_baseline_over_candidate": descriptive([row["baseline_over_candidate"] for row in per_block]),
            "paired_speedup_baseline_repeat_over_candidate": descriptive([row["baseline_repeat_over_candidate"] for row in per_block]),
            "same_code_noise_baseline_over_repeat": descriptive([row["baseline_over_baseline_repeat"] for row in per_block]),
            "same_full_state_digest": len(digests) == 1,
            "same_kernel_input_digest": len(input_digests) == 1 if mode == "kernel" else None,
            "state_digest": next(iter(digests)) if len(digests) == 1 else None,
            "blocks": per_block}


def run_process(command, log_path, timeout):
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, env={**os.environ, **THREAD_ENV}, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"Benchmark process failed ({completed.returncode}); see {log_path}")


def parent(args):
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    if baseline == candidate:
        raise ValueError("Baseline and candidate must use separate source/build roots")
    roots = {"baseline": baseline, "candidate": candidate, "baseline-repeat": baseline}
    for root in (baseline, candidate):
        if not (root / "_tiemu_ffi.py").is_file():
            raise ValueError(f"Missing source root: {root}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    phase_script = script.with_name("benchmark_simulation.py")
    kernel, phase = {}, {}
    for mode, records in (("kernel", kernel), ("phase", phase)):
        for block in range(args.blocks):
            for name in ORDERS[block % len(ORDERS)]:
                destination = output / f"{mode}-{block:02d}-{name}.json"
                if mode == "kernel":
                    command = [sys.executable, str(script), "--worker", "--source-root", str(roots[name]),
                               "--configuration", name, "--block", str(block), "--samples", str(args.samples),
                               "--target-cell-updates", str(args.target_cell_updates), "--max-calls", str(args.max_calls),
                               "--worker-output", str(destination)]
                else:
                    command = [sys.executable, str(phase_script), "--source-root", str(roots[name]),
                               "--phase", "inorganic", "--generations", "10000", "--population", "100",
                               "--backend", "cpp", "--repeats", str(args.samples), "--output", str(destination)]
                run_process(command, destination.with_suffix(".log"), args.process_timeout)
                record = json.loads(destination.read_text(encoding="utf-8"))
                if Path(record["core_file"]).resolve().parent != roots[name]:
                    raise RuntimeError("Reported native core escaped its selected source root")
                sample_count = len(record["cases"][0]["samples"]) if mode == "kernel" else len(record["seconds"])
                if sample_count != args.samples:
                    raise AssertionError("Child did not return every requested timing sample")
                records[(block, name)] = record
            print(json.dumps({"completed": mode, "block": block + 1, "blocks": args.blocks}), flush=True)
    hashes = {name: {kernel[(block, name)]["core_sha256"] for block in range(args.blocks)}
              for name in CONFIGURATIONS}
    if any(len(values) != 1 for values in hashes.values()) or hashes["baseline"] != hashes["baseline-repeat"]:
        raise AssertionError("Native binaries changed during the experiment or noise control used different code")
    kernel_rows = [summarize_case(kernel, f"grid{grid}-steps{steps}", args.blocks, "kernel") for grid, steps in CASES]
    phase_row = summarize_case(phase, "actual-inorganic-grid10-generations10000", args.blocks, "phase")
    phase_row["same_legacy"] = len({digest_json(outcome) for record in phase.values()
                                    for outcome in record["outcomes"]}) == 1
    accepted = (all(row["same_full_state_digest"] and row["same_kernel_input_digest"] for row in kernel_rows)
                and phase_row["same_full_state_digest"] and phase_row["same_legacy"])
    estimated_row_bytes = kernel[(0, "candidate")]["estimated_table_bytes_per_cell"]
    report = {
        "schema_version": 1, "baseline": str(baseline), "candidate": str(candidate),
        "blocks": args.blocks, "samples_per_process_case": args.samples,
        "configuration_orders": [list(ORDERS[block % len(ORDERS)]) for block in range(args.blocks)],
        "commits": {name: sorted({kernel[(block, name)]["commit"] for block in range(args.blocks)}) for name in CONFIGURATIONS},
        "core_sha256": {name: next(iter(values)) for name, values in hashes.items()},
        "same_checked_results": accepted, "kernel_cases": kernel_rows, "actual_phase": phase_row,
        "estimated_candidate_table_bytes": {str(grid): grid * grid * estimated_row_bytes for grid in (1, 2, 10, 32)},
        "memory_note": "Estimated NeighborRow storage only, not measured memory/RSS; excludes vector/allocator overhead.",
        "scope": "One hosted CPU runner, sequential fresh processes, all six configuration permutations per cycle. "
                 "Baseline-repeat uses the identical baseline binary. Headline ratios use every paired block median; "
                 "raw per-sample timings/digests are retained. Kernel timing includes per-call table construction and "
                 "ctypes boundary/loop overhead; inputs/outputs are preallocated. Actual phase uses the same harness "
                 "for both source roots, grid10 x 10000 generations, CPU1, no LLM/display/log/keyboard/exports, with "
                 "headless <=100-step segments. It includes initialization, Python/native movement and final legacy "
                 "reduction; excludes imports, simulation setup, warm-up, final state hashing and close. Actual phase "
                 "checks complete legacy, Python/NumPy/Torch RNG and generation, but exposes no final cell arrays; "
                 "kernel cases independently hash every activity/stability/MT byte. Not full-world or local-M1 proof.",
        "decision_note": "Exact results are required, not sufficient for adoption. Compare candidate paired distributions "
                         "with same-code noise, especially actual 100-step chunks and single-step regressions. Macro "
                         "activation must be proven separately by workflow compiler logs; no fastest-sample selection.",
    }
    write_json(output / "summary.json", report)
    print(json.dumps({"same_checked_results": accepted, "summary": str(output / "summary.json"),
                      "actual_phase_paired_speedup": phase_row["paired_speedup_baseline_over_candidate"]["median"],
                      "actual_phase_same_code_noise": phase_row["same_code_noise_baseline_over_repeat"]["median"]}), flush=True)
    if not accepted:
        raise SystemExit("Inorganic results differ; do not accept candidate performance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--target-cell-updates", type=int, default=250000)
    parser.add_argument("--max-calls", type=int, default=4096)
    parser.add_argument("--process-timeout", type=int, default=180)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--configuration", choices=CONFIGURATIONS)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.blocks not in (6, 12) or not 3 <= args.samples <= 9:
        parser.error("Use 6 or 12 balanced blocks, with 3..9 samples")
    if not 1 <= args.target_cell_updates <= 1000000 or not 1 <= args.max_calls <= 10000:
        parser.error("Cell update target must be 1..1000000 and call cap 1..10000")
    if not 1 <= args.process_timeout <= 600 or not 0 <= args.block < 12:
        parser.error("Process timeout must be 1..600 seconds and block index 0..11")
    if args.worker:
        if args.source_root is None or args.configuration is None or args.worker_output is None:
            parser.error("Worker requires --source-root, --configuration and --worker-output")
        kernel_worker(args)
    else:
        if args.baseline is None or args.candidate is None or args.output_dir is None:
            parser.error("Parent requires --baseline, --candidate and --output-dir")
        parent(args)


if __name__ == "__main__":
    main()
