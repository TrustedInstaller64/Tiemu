"""Run the unchanged upstream commit with this runner's numerical libraries.

The checked-in JSON fixtures record the original Apple Silicon run. They are
useful evidence, but are not portable oracles: upstream uses an unstable argsort
at tied election boundaries, and CPU float32 implementations can differ.
Neither the upstream code nor the exact discrete/RNG assertions are relaxed.
"""
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "691c6e0023a83513ad3cd386cfee2198317104d3"


def replay_environment():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


@lru_cache(maxsize=1)
def _baseline_checkout():
    archive = subprocess.run(
        ["git", "archive", "--format=tar", BASELINE_COMMIT], cwd=ROOT,
        capture_output=True, timeout=30)
    if archive.returncode:
        error = archive.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Cannot export frozen upstream {BASELINE_COMMIT}. The equivalence "
            f"tests require its Git history; CI checkout must use fetch-depth: 0.\n{error}")
    checkout = tempfile.TemporaryDirectory(prefix="tiemu-frozen-upstream-")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as source:
            source.extractall(checkout.name, filter="data")
    except BaseException:
        checkout.cleanup()
        raise
    # The cached TemporaryDirectory owns cleanup until this test process exits.
    return checkout


@lru_cache(maxsize=None)
def _reference_json(script_name, arguments):
    if script_name not in {"replay_world.py", "replay_phases.py"}:
        raise ValueError(f"Unknown upstream replay script: {script_name}")
    baseline = _baseline_checkout()
    with tempfile.TemporaryDirectory(prefix="tiemu-reference-run-") as working_dir:
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "tests" / script_name),
             "--root", baseline.name, "--backend", "python", *arguments],
            cwd=working_dir, env=replay_environment(), capture_output=True,
            text=True, encoding="utf-8", timeout=60)
        if result.returncode:
            raise AssertionError(
                f"Frozen upstream replay failed ({script_name}, {arguments}):\n"
                f"{result.stdout}\n{result.stderr}")
        if list(Path(working_dir).rglob("*.pth")):
            raise AssertionError("Frozen upstream replay unexpectedly exported a model")
    # Cache serialized output so tests cannot mutate the next case's oracle.
    json.loads(result.stdout)
    return result.stdout


def reference_result(script_name, *arguments):
    """Return independent expected state from the fixed commit on this platform."""
    return json.loads(_reference_json(script_name, tuple(arguments)))
