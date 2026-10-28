import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_inorganic_next_stops_again_after_one_generation():
    from types import SimpleNamespace
    from simulation import AeonEvolution
    sim = object.__new__(AeonEvolution)
    pauses = []
    def handle_commands():
        pauses.append(sim.generation)
        sim.debugger.paused = False
        if len(pauses) == 1:
            sim.debugger.last_command = 'next'
        else:
            sim.debugger.stop_requested = True
    sim.debugger = SimpleNamespace(paused=True, stop_requested=False, last_command='',
                                   poll=lambda: None, handle_commands=handle_commands)
    sim.display_manager = SimpleNamespace(update_and_display_progress=lambda *a: None)
    assert sim._run_inorganic_phase(3) is None
    assert pauses == [0, 1]

def test_headless_cli_finishes_with_bounded_phases(tmp_path):
    run = subprocess.run([sys.executable, str(ROOT/'main.py'), '--disable-llm', '--no-interactive',
                          '--backend', 'python', '--device', 'cpu', '--seed', '42', '--smoke-test',
                          '--output-dir', str(tmp_path)], cwd=tmp_path, capture_output=True, encoding='utf-8',
                         env={**os.environ, 'PYTHONIOENCODING': 'ascii'}, timeout=40)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path/'world.json').exists()
    assert (tmp_path/'blueprint_policy.pth').exists()
    assert 'Traceback' not in run.stderr
    assert '(Loss:' not in run.stdout

def test_cli_errors_have_nonzero_status(tmp_path):
    run = subprocess.run([sys.executable, str(ROOT/'main.py'), '--device', 'bad-device'],
                         cwd=tmp_path, capture_output=True, encoding='utf-8')
    assert run.returncode != 0
    assert not list(tmp_path.glob('*.pth'))


def test_incomplete_snapshot_cannot_silently_start_a_new_world(tmp_path):
    snapshot = tmp_path/'world.json'
    original = '{"generation": 1000}'
    snapshot.write_text(original)
    run = subprocess.run([sys.executable, str(ROOT/'main.py'), '--load-save', str(snapshot),
                          '--output-dir', str(tmp_path), '--smoke-test'],
                         capture_output=True, encoding='utf-8', timeout=10)
    assert run.returncode != 0
    assert snapshot.read_text() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ['world.json']


def test_cpu_thread_limit_rejects_zero():
    import pytest
    from main import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--torch-threads', '0'])
    assert build_parser().parse_args([]).torch_threads == 1


def test_sigterm_saves_and_exits_without_a_terminal(tmp_path):
    import os
    import signal
    import time
    import pytest
    if os.name == 'nt':
        pytest.skip('TerminateProcess is not graceful SIGTERM on Windows')
    process = subprocess.Popen([sys.executable, '-u', str(ROOT/'main.py'), '--disable-llm',
                                '--no-interactive', '--backend', 'cpp', '--device', 'cpu',
                                '--output-dir', str(tmp_path)], cwd=tmp_path,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    try:
        # Read the CLI readiness line; no arbitrary sleep before sending SIGTERM.
        import selectors
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 20
        output = b''
        while '计算后端'.encode() not in output:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and selector.select(timeout=remaining), 'CLI failed to start'
            chunk = os.read(process.stdout.fileno(), 4096)
            assert chunk, output.decode(errors='replace')
            output += chunk
        selector.close()
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=20)
        assert process.returncode == 143, out + err
        assert (tmp_path/'world.json').exists()
        assert (tmp_path/'blueprint_policy.pth').exists()
        assert not list(tmp_path.glob('*.tmp'))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
