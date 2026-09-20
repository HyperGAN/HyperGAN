"""One in-flight immutable preview, rendered and published by a CPU process."""
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread
import tempfile
import time

from .snapshot_renderer import render_snapshot
from .run_controller import FatalExecutionError


class PreviewError(RuntimeError):
    def __init__(self, error, step, identity):
        super().__init__(f'{type(error).__name__}: {error}'[:1000])
        self.preview_context = {'step': step, 'identity': dict(identity)}
        self.__cause__ = error


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

    def submit(self, temporary, descriptor, identity, step, run_dir, keep, *, snapshot_state=None):
        if self.busy:
            raise RuntimeError('Preview worker already has an outstanding snapshot')
        self._cancel.clear()
        future = Future()
        self._future = future
        self._deadline = time.monotonic() + self.timeout
        self._source = {'step': step, 'identity': dict(identity)}

        def render():
            nonlocal snapshot_state
            owned_temporary, error = temporary, None
            try:
                if snapshot_state is not None:
                    owned_temporary = tempfile.TemporaryDirectory(prefix='.preview-', dir=run_dir)
                directory = Path(owned_temporary.name)
                frozen_descriptor = descriptor
                if snapshot_state is not None:
                    from .preview_snapshot import write_snapshot
                    if self._cancel.is_set():
                        raise RuntimeError('Preview cancelled before snapshot persistence')
                    frozen_descriptor = write_snapshot(snapshot_state, directory / 'snapshot.pt',
                        cancellation_event=self._cancel, deadline=self._deadline)
                    snapshot_state = None  # Release captured tensors before the renderer loads its copy.
                if self._cancel.is_set():
                    raise RuntimeError('Preview cancelled before rendering')
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Preview snapshot persistence deadline exceeded')
                result = render_snapshot(directory / 'snapshot.pt', frozen_descriptor, identity, step,
                    directory / 'preview.json', timeout=remaining, publish_run_dir=run_dir,
                    keep=keep, cancellation_event=self._cancel)
            except BaseException as failure:
                error = failure
            try:
                if owned_temporary is not None:
                    owned_temporary.cleanup()
            except BaseException as cleanup:
                if error is None:
                    error = cleanup
                elif hasattr(error, 'add_note'):
                    error.add_note(f'Preview cleanup also failed: {cleanup}')
            if error is not None:
                if not isinstance(error, Exception) or isinstance(error, FatalExecutionError):
                    error.preview_context = {'step': step, 'identity': dict(identity)}
                else:
                    error = PreviewError(error, step, identity)
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
        if future is None:
            return None
        if not future.done() and time.monotonic() >= self._deadline + 8:
            self._cancel.set()
            failure = TimeoutError('Preview persistence/rendering exceeded its deadline and cleanup grace')
            failure.preview_context = dict(self._source)
            raise failure
        if not wait and not future.done():
            return None
        try:
            # Waiting is reserved for terminal cleanup. CPUWorkerService owns
            # finite startup/command/total deadlines and worker reaping.
            return future.result(timeout=max(0, self._deadline + 8 - time.monotonic()) if wait else 0)
        except TimeoutError as error:
            if not future.done():
                self._cancel.set()
                failure = TimeoutError('Preview persistence/rendering exceeded its deadline and cleanup grace')
                failure.preview_context = dict(self._source)
                raise failure from error
            return future.result()  # A result may have arrived as the wait timed out.
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
