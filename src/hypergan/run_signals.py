"""Bounded cooperative termination for the controller's main thread."""
import os
import signal
import threading


class GracefulStop:
    """First signal finishes an update; another signal or deadline forces exit.

    No serialization, logging, or checkpoint work runs in a signal handler.
    SIGKILL and forced exits recover the previous atomic checkpoint pointer.
    Embedded training outside the main thread retains the host's signal policy.
    """
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.signum = None
        self.previous = {}
        self.timer = None

    @property
    def reason(self):
        return signal.Signals(self.signum).name if self.signum is not None else None

    def _receive(self, signum, frame):
        if self.signum is not None:
            os._exit(128 + signum)
        self.signum = signum
        self.timer = threading.Timer(self.timeout, os._exit, args=(128 + signum,))
        self.timer.daemon = True
        self.timer.start()

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous[signum] = signal.signal(signum, self._receive)
        return self

    def __exit__(self, *unused):
        if self.timer is not None:
            self.timer.cancel()
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
