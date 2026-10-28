"""Measure the project's single-state training paths, including device copies.

Example (Apple Silicon, outside GPU-restricting sandboxes)::

    python benchmarks/benchmark_networks.py --devices cpu mps --threads 1 \
        --steps 100 --repeats 5 --output network-results.json

No external model weights or world simulation are used. Run different CPU
thread counts in separate processes. Results are not power measurements.
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from models import ActionPolicyNetwork, HybridGuideNetwork, ValueNetwork


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps", "cuda"], default=["cpu"])
    parser.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU intra-op threads; inter-op stays at 1")
    parser.add_argument("--steps", type=positive_int, default=100, help="training updates per measured repeat")
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=positive_int, default=20, help="unmeasured training updates per case/device")
    parser.add_argument("--output", type=Path, help="also write JSON to this path; stdout always receives the final result")
    return parser


def hardware_name():
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def make_case(kind, device):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    loss_fn = torch.nn.MSELoss()
    if kind == "guide_train":
        model = HybridGuideNetwork(49, 128, 12).to(device).train()
        models = [model]
        initial = [{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        state_np = rng.normal(size=49).astype(np.float64)
        target_np = np.full(12, 0.5, dtype=np.float64)

        def step():
            state = torch.from_numpy(state_np).float().to(device)
            target = torch.from_numpy(target_np).float().to(device)
            optimizer.zero_grad()
            predicted = model(state)
            imitation_loss = loss_fn(predicted, target)
            loss = imitation_loss + (-imitation_loss * 0.5) * 0.1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            predicted.detach().cpu().numpy()
            return loss

        shape = [49]
    else:
        action = ActionPolicyNetwork(24, 32).to(device).train()
        value = ValueNetwork(24, 32).to(device).train()
        models = [action, value]
        initial = [{k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for model in models]
        action_optimizer = torch.optim.Adam(action.parameters(), lr=0.0025)
        value_optimizer = torch.optim.Adam(value.parameters(), lr=0.005)
        state_np = rng.normal(size=24).astype(np.float64)

        def step():
            state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            probs = torch.softmax(action(state), dim=-1)
            chosen_action = torch.distributions.Categorical(probs).sample().item()
            state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            state_value = value(state)
            reward = 1.0
            advantage = reward - state_value.item()
            value_optimizer.zero_grad()
            value_loss = loss_fn(state_value, torch.tensor([[reward]], dtype=torch.float32, device=device))
            value_loss.backward(retain_graph=True)
            value_optimizer.step()
            probs = torch.softmax(action(state), dim=-1)
            distribution = torch.distributions.Categorical(probs)
            index = torch.tensor([chosen_action], dtype=torch.long, device=device)
            log_prob = distribution.log_prob(index)
            action_optimizer.zero_grad()
            actor_loss = -log_prob * advantage
            actor_loss.backward()
            action_optimizer.step()
            return actor_loss + value_loss

        shape = [1, 24]
    return models, initial, step, shape


def measure_case(kind, device, args):
    models, initial, step, shape = make_case(kind, device)
    for _ in range(args.warmup):
        last_loss = step()
    synchronize(device)
    durations = []
    for _ in range(args.repeats):
        synchronize(device)
        started = time.perf_counter()
        for _ in range(args.steps):
            last_loss = step()
        synchronize(device)
        durations.append(time.perf_counter() - started)
    current = [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in models]
    finite = all(bool(torch.isfinite(v).all()) for state in current for v in state.values())
    changed = [any(not torch.equal(old[k], new[k]) for k in old) for old, new in zip(initial, current)]
    final_loss = float(last_loss.detach().cpu())
    if not finite or not all(changed) or not math.isfinite(final_loss):
        raise RuntimeError(f"{kind}/{device}: invalid or unchanged training weights/loss")
    return {
        "case": kind,
        "device": device,
        "input_shape": shape,
        "parameter_counts": [sum(p.numel() for p in model.parameters()) for model in models],
        "repeat_seconds": durations,
        "median_ms_per_step": statistics.median(durations) * 1000 / args.steps,
        "weights_finite": finite,
        "weights_changed": changed,
        "last_loss": final_loss,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            parser.error("MPS is unavailable; select --devices cpu or run on a Mac with GPU access")
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; select --devices cpu or install a supported CUDA runtime")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    report = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "hardware": hardware_name(),
        "logical_cpus": os.cpu_count(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "dtype": "float32",
        "seed": 42,
        "warmup_steps": args.warmup,
        "steps_per_repeat": args.steps,
        "repeats": args.repeats,
        "scope": "Bounded network updates only; includes original host/device copies and item() dependencies, excludes world simulation and logging.",
        "network_weights": "Fresh seeded initialization; no external checkpoint.",
        "results": [],
        "comparisons": [],
    }
    for kind in ("guide_train", "action_value_train"):
        for device in dict.fromkeys(args.devices):
            print(f"Measuring {kind} on {device}...", file=sys.stderr, flush=True)
            report["results"].append(measure_case(kind, device, args))
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        rows = {r["device"]: r for r in report["results"] if r["case"] == kind}
        if "cpu" in rows:
            for device in rows.keys() - {"cpu"}:
                cpu_ms = rows["cpu"]["median_ms_per_step"]
                device_ms = rows[device]["median_ms_per_step"]
                report["comparisons"].append({
                    "case": kind,
                    f"{device}_over_cpu_time": device_ms / cpu_ms,
                    f"{device}_speedup": cpu_ms / device_ms,
                })
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
