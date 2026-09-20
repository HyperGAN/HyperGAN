"""A single bounded background read slot for optional live control polling."""
import threading
import time


class BackgroundPoll:
    """Cache one read outcome; ordinary poll never waits for filesystem work.

    A final explicit control boundary can use read_now(), which waits for an
    in-flight scan and then reads fresh state. close(wait=False) abandons optional
    observation promptly; its daemon exits once any stalled OS read returns.
    """
    def __init__(self, read, *, interval=0.25):
        self.read = read
        self.interval = interval
        self.condition = threading.Condition()
        self.requested = self.busy = self.stopped = False
        self.ready = False
        self.result = self.error = None
        self.next_read = 0.0
        self.thread = threading.Thread(target=self._run, name='hypergan-control-read', daemon=True)
        self.thread.start()

    def poll(self, *, schedule=True):
        with self.condition:
            ready, result, error = self.ready, self.result, self.error
            if ready:
                self.ready = False
                self.result = self.error = None
            now = time.monotonic()
            if schedule and not ready and not self.stopped and not self.busy and not self.requested and now >= self.next_read:
                self.requested = True
                self.next_read = now + self.interval
                self.condition.notify_all()
        if error is not None:
            raise error
        return ready, result

    def read_now(self):
        with self.condition:
            while self.busy or self.requested:
                self.condition.wait()
            self.ready = False
            self.result = self.error = None
            self.next_read = time.monotonic() + self.interval
        return self.read()

    def close(self, *, wait=False):
        with self.condition:
            self.stopped = True
            self.requested = False
            self.condition.notify_all()
        if wait:
            self.thread.join()

    def _run(self):
        while True:
            with self.condition:
                while not self.requested and not self.stopped:
                    self.condition.wait()
                if self.stopped:
                    return
                self.requested = False
                self.busy = True
            result = error = None
            try:
                result = self.read()
            except BaseException as exc:
                error = exc
            with self.condition:
                self.result, self.error = result, error
                self.ready = True
                self.busy = False
                self.condition.notify_all()
