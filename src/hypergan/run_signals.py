"""Bounded cooperative termination for the controller's main thread."""
import os
import signal
import threading
import time


class GracefulStop:
    """First signal finishes an update; another signal or deadline forces exit.

    No serialization, logging, thread spawning, or checkpoint work runs in a
    signal handler. SIGKILL and forced exits recover the previous atomic
    checkpoint pointer. Embedded training outside the main thread retains the
    host's signal policy.
    """
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.signum = None
        self.previous = {}
        self._deadline = None
        self._stop_event = None
        self._watchdog = None

    @property
    def reason(self):
        return signal.Signals(self.signum).name if self.signum is not None else None

    def _watch(self):
        # Poll a monotonic deadline set by the signal handler so the handler
        # itself never spawns threads, acquires locks, or allocates timers.
        poll = min(0.05, max(0.005, float(self.timeout) / 10.0)) if self.timeout and self.timeout > 0 else 0.01
        stop_event = self._stop_event
        while stop_event is not None and not stop_event.is_set():
            deadline = self._deadline
            signum = self.signum
            if deadline is not None and signum is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    os._exit(128 + signum)
                stop_event.wait(min(poll, remaining))
            else:
                stop_event.wait(poll)

    def _receive(self, signum, frame):
        if self.signum is not None:
            os._exit(128 + signum)
        self._deadline = time.monotonic() + self.timeout
        self.signum = signum

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self._deadline = None
            self._stop_event = threading.Event()
            self._watchdog = threading.Thread(
                target=self._watch,
                name="hypergan-graceful-stop-watchdog",
                daemon=True,
            )
            self._watchdog.start()
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous[signum] = signal.signal(signum, self._receive)
        return self

    def __exit__(self, *unused):
        if self._stop_event is not None:
            self._stop_event.set()
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        self.previous.clear()
        if self._watchdog is not None:
            self._watchdog.join(timeout=0.5)
            self._watchdog = None
        self._stop_event = None
