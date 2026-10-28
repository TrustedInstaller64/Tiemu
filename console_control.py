"""Bounded console input for one caller; no keyboard hooks or reader threads."""

import codecs
import os
import sys
import time


class ConsoleInput:
    def __init__(self, interactive=None, *, stream=None, output=None):
        self.stream = sys.stdin if stream is None else stream
        self.output = sys.stdout if output is None else output
        self.interactive = False
        self.eof = False
        self._fd = None
        self._settings = None
        self._windows = os.name == "nt"
        self._special_key = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        if interactive is False:
            return
        try:
            if self.stream is None or not self.stream.isatty():
                return
            self._fd = self.stream.fileno()
            if self._windows:
                import msvcrt
                self._msvcrt = msvcrt
            else:
                import select
                import termios
                import tty
                self._select = select
                self._termios = termios
                try:
                    self._settings = termios.tcgetattr(self._fd)
                    # Keep ISIG (Ctrl+C) and pending input; never enter raw mode.
                    tty.setcbreak(self._fd, termios.TCSANOW)
                except termios.error:
                    self.close()
                    return
            self.interactive = True
        except (AttributeError, ImportError, OSError, ValueError):
            self.close()

    def read_char(self, timeout=0.0):
        """Return one Unicode character, or None after a bounded wait/EOF."""
        if not self.interactive:
            return None
        deadline = time.monotonic() + max(0.0, timeout)
        try:
            # Bound even a stream of function keys or malformed UTF-8.
            for _ in range(64):
                if self._windows:
                    if self._msvcrt.kbhit():
                        char = self._msvcrt.getwch()
                        if self._special_key:
                            self._special_key = False
                            continue
                        if char in ("\x00", "\xe0"):
                            self._special_key = True
                            continue
                        return char
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    time.sleep(min(0.01, remaining))
                else:
                    remaining = max(0.0, deadline - time.monotonic())
                    ready, _, _ = self._select.select([self._fd], [], [], remaining)
                    if not ready:
                        return None
                    data = os.read(self._fd, 1)
                    if not data:
                        self.eof = True
                        self.close()
                        return None
                    char = self._decoder.decode(data)
                    if char:
                        return char
        except (OSError, ValueError):
            # A closed terminal or a Docker attach disconnect disables input.
            self.eof = True
            self.close()
        return None

    def write(self, text):
        try:
            self.output.write(text)
            self.output.flush()
        except (AttributeError, OSError, ValueError):
            pass

    def close(self):
        self.interactive = False
        if self._settings is not None:
            settings, self._settings = self._settings, None
            try:
                # The stream may already be closed; don't restore a reused fd.
                if self.stream.fileno() == self._fd and self.stream.isatty():
                    self._termios.tcsetattr(self._fd, self._termios.TCSANOW, settings)
            except (AttributeError, OSError, ValueError, self._termios.error):
                pass
