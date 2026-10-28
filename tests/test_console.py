"""Console contracts; POSIX tests use isolated PTYs, never the user's terminal."""

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from debugger import Debugger
from console_control import ConsoleInput


class SimulationFixture:
    def __init__(self):
        self.growth_factor = 0.35
        self.loaded_path = None

    def save_simulation_state(self, path):
        Path(path).write_text(json.dumps({"growth_factor": self.growth_factor}))

    def load_simulation_state(self, path):
        self.loaded_path = path


class ConsoleTests(unittest.TestCase):
    def make_debugger(self, sim=None, interactive=None):
        return Debugger(sim or SimulationFixture(), interactive=interactive)

    @contextlib.contextmanager
    def terminal(self):
        import pty
        import termios

        master, slave = pty.openpty()
        stream = os.fdopen(slave, "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave)
        output = io.StringIO()
        try:
            with mock.patch("sys.stdin", stream), mock.patch("sys.stdout", output):
                yield master, stream, original, output
        finally:
            stream.close()
            os.close(master)

    def test_non_tty_disables_input_and_does_not_spawn_a_reader(self):
        before = set(threading.enumerate())
        with mock.patch("sys.stdin", io.StringIO("p\nstop\n")):
            debugger = self.make_debugger(interactive=True)
            try:
                debugger.poll()
                self.assertFalse(debugger.paused)
                self.assertFalse(debugger.stop_requested)
                self.assertFalse(debugger.interactive)
                self.assertEqual(set(threading.enumerate()), before)
            finally:
                debugger.close()

    def test_stop_request_retains_signal_and_breaks_paused_wait(self):
        with mock.patch("sys.stdin", io.StringIO()):
            debugger = self.make_debugger(interactive=False)
            debugger.paused = True
            debugger.request_stop(signal.SIGTERM)
            debugger.request_stop()
            debugger.handle_commands()
            self.assertTrue(debugger.stop_requested)
            self.assertEqual(debugger.stop_signal, signal.SIGTERM)
            debugger.close()

    def test_headless_manual_pause_cannot_wait_forever_for_input(self):
        with mock.patch("sys.stdin", io.StringIO()):
            debugger = self.make_debugger(interactive=False)
            try:
                debugger.paused = True
                debugger.handle_commands()
                self.assertFalse(debugger.paused)
                self.assertFalse(debugger.stop_requested)
            finally:
                debugger.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_posix_hotkey_and_queued_commands_have_one_input_owner(self):
        sim = SimulationFixture()
        with self.terminal() as (master, _stream, _original, _output):
            debugger = self.make_debugger(sim)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    saved = str(Path(tmp) / "snapshot.json")
                    os.write(master, ("pset growth_factor 0.5\nsave " + saved + "\nn\n").encode())
                    debugger.poll()
                    self.assertTrue(debugger.paused)
                    debugger.handle_commands()
                    self.assertFalse(debugger.paused)
                    self.assertEqual(debugger.last_command, "next")
                    self.assertEqual(json.loads(Path(saved).read_text()), {"growth_factor": 0.5})
            finally:
                debugger.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_posix_cbreak_preserves_signals_and_close_restores_terminal(self):
        import termios

        with self.terminal() as (_master, stream, original, _output):
            debugger = self.make_debugger()
            changed = termios.tcgetattr(stream.fileno())
            self.assertTrue(changed[3] & termios.ISIG)
            self.assertFalse(changed[3] & termios.ICANON)
            debugger.close()
            debugger.close()
            restored = termios.tcgetattr(stream.fileno())
            # Darwin sets PENDIN when returning to canonical input; it is a
            # transient kernel retype request, not a lost terminal setting.
            restored[3] &= ~getattr(termios, "PENDIN", 0)
            original[3] &= ~getattr(termios, "PENDIN", 0)
            self.assertEqual(restored, original)

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_explicit_no_interactive_does_not_modify_a_tty(self):
        import termios

        with self.terminal() as (master, stream, original, _output):
            debugger = self.make_debugger(interactive=False)
            try:
                os.write(master, b"p\n")
                debugger.poll()
                self.assertFalse(debugger.paused)
                self.assertEqual(termios.tcgetattr(stream.fileno()), original)
            finally:
                debugger.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_unicode_save_command_and_backspace_survive_terminal_input(self):
        sim = SimulationFixture()
        saved = []
        sim.save_simulation_state = saved.append
        with self.terminal() as (master, _stream, _original, _output):
            debugger = self.make_debugger(sim)
            try:
                debugger.paused = True
                os.write(master, "save 存档错\x7f.json\nc\n".encode())
                debugger.handle_commands()
                self.assertEqual(saved, ["存档.json"])
                self.assertFalse(debugger.paused)
            finally:
                debugger.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_stop_command_leaves_command_loop(self):
        with self.terminal() as (master, _stream, _original, _output):
            debugger = self.make_debugger()
            try:
                debugger.paused = True
                os.write(master, b"stop\n")
                debugger.handle_commands()
                self.assertTrue(debugger.stop_requested)
                self.assertIsNone(debugger.stop_signal)
            finally:
                debugger.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_stop_request_interrupts_paused_wait_without_enter(self):
        with self.terminal() as (_master, _stream, _original, _output):
            debugger = self.make_debugger()
            timer = threading.Timer(0.05, debugger.request_stop)
            try:
                debugger.paused = True
                timer.start()
                started = time.monotonic()
                debugger.handle_commands()
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertTrue(debugger.stop_requested)
            finally:
                debugger.close()
                timer.join()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_ctrl_d_does_not_leave_simulation_stuck_paused(self):
        with self.terminal() as (master, _stream, _original, _output):
            debugger = self.make_debugger()
            try:
                debugger.paused = True
                os.write(master, b"\x04")
                debugger.handle_commands()
                self.assertFalse(debugger.paused)
                self.assertFalse(debugger.interactive)
            finally:
                debugger.close()

    def test_windows_adapter_returns_unicode_and_skips_function_key_codes(self):
        # Branch contract only: an actual Windows console still needs Windows CI/manual validation.
        keys = iter(["\xe0", "H", "中", "p"])
        windows = SimpleNamespace(kbhit=lambda: True, getwch=lambda: next(keys))
        stream = SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)
        with mock.patch("console_control.os.name", "nt"), mock.patch.dict("sys.modules", {"msvcrt": windows}):
            console = ConsoleInput(stream=stream, output=io.StringIO())
            try:
                self.assertEqual(console.read_char(), "中")
                self.assertEqual(console.read_char(), "p")
            finally:
                console.close()

    def test_pipe_is_not_treated_as_a_terminal(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
            try:
                os.write(write_fd, b"p\n")
                console = ConsoleInput(stream=stream)
                try:
                    self.assertFalse(console.interactive)
                    self.assertIsNone(console.read_char(timeout=0.1))
                finally:
                    console.close()
            finally:
                os.close(write_fd)

    @unittest.skipIf(os.name == "nt", "POSIX terminal error handling")
    def test_terminal_disappearing_during_initialization_disables_input(self):
        import termios

        stream = SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)
        with mock.patch("termios.tcgetattr", side_effect=termios.error(25, "not a tty")):
            console = ConsoleInput(stream=stream)
            self.assertFalse(console.interactive)
            console.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_stop_command_is_case_insensitive_like_existing_commands(self):
        with self.terminal() as (master, _stream, _original, _output):
            debugger = self.make_debugger()
            try:
                debugger.paused = True
                os.write(master, b"STOP\nc\n")
                debugger.handle_commands()
                self.assertTrue(debugger.stop_requested)
            finally:
                debugger.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_terminal_hangup_disables_reading_instead_of_spinning(self):
        import pty

        master, slave = pty.openpty()
        with os.fdopen(slave, "r", encoding="utf-8") as stream:
            console = ConsoleInput(stream=stream, output=io.StringIO())
            os.close(master)
            try:
                self.assertIsNone(console.read_char(timeout=0.05))
                self.assertTrue(console.eof)
                self.assertFalse(console.interactive)
                started = time.monotonic()
                self.assertIsNone(console.read_char(timeout=5))
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                console.close()

    @unittest.skipIf(os.name == "nt", "POSIX PTY coverage")
    def test_status_reads_mutation_rate_from_population_manager(self):
        sim = SimpleNamespace(
            population=[], generation=12, total_generations=100,
            population_manager=SimpleNamespace(mutation_rate=0.25),
            stagnation_manager=SimpleNamespace(long_term_stagnation_counter=1, baie_stagnation_counter=0),
            reincarnator=None,
        )
        with self.terminal() as (master, _stream, _original, _output):
            debugger = self.make_debugger(sim)
            try:
                debugger.paused = True
                os.write(master, b"status\nc\n")
                with self.assertLogs("OmphalosLogger", level="INFO") as captured:
                    debugger.handle_commands()
                self.assertTrue(any("0.2500" in line for line in captured.output))
                self.assertFalse(debugger.paused)
            finally:
                debugger.close()


if __name__ == "__main__":
    unittest.main()
