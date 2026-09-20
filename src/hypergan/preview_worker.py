"""One in-flight immutable preview, rendered and published by a CPU process."""
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread

from .snapshot_renderer import render_snapshot


class PreviewWorker:
    """The supervising thread handles no live trainer objects or global RNG.

    Capture occurs at the update boundary before submit. A busy slot rejects new
    work before capture; callers poll the completed result at later boundaries.
    The renderer's broker bounds execution and reaps it after coordinator death.
    """
    def __init__(self, timeout=60):
        self.timeout = timeout
        self._future = None
        self._thread = None
        self._cancel = Event()

    @property
    def busy(self):
        return self._future is not None

    def submit(self, temporary, descriptor, identity, step, run_dir, keep):
        if self.busy:
            raise RuntimeError('Preview worker already has an outstanding snapshot')
        future = Future()
        self._future = future

        def render():
            try:
                with temporary:
                    directory = Path(temporary.name)
                    result = render_snapshot(directory / 'snapshot.pt', descriptor, identity, step,
                        directory / 'preview.json', timeout=self.timeout, publish_run_dir=run_dir,
                        keep=keep, cancellation_event=self._cancel)
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(result)

        try:
            self._thread = Thread(target=render, name='hypergan-preview-supervisor', daemon=True)
            self._thread.start()
        except BaseException:
            self._future = None
            raise

    def poll(self, *, wait=False):
        future = self._future
        if future is None or (not wait and not future.done()):
            return None
        try:
            # Waiting is reserved for terminal cleanup. CPUWorkerService owns
            # finite startup/command/total deadlines and worker reaping.
            return future.result()
        finally:
            if future.done():
                self._future = None

    def abort(self):
        if not self.busy:
            return False
        self._cancel.set()
        self._thread.join(8)
        if self._thread.is_alive():
            raise RuntimeError('Preview dispatcher has not completed worker cleanup')
        self._future = None
        return True
