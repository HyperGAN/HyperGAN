"""Preview admission is bounded and never waits for CPU rendering mid-update."""
import tempfile
import json
import os
import time
from pathlib import Path
from threading import Event, Timer

import pytest

from hypergan.preview_worker import PreviewWorker


def test_preview_slot_is_nonblocking_and_retains_source_identity(tmp_path, monkeypatch):
    import hypergan.preview_worker as module
    entered, release = Event(), Event()
    identity = {'run_id': 'run', 'attempt_id': 'attempt', 'sample_sequence': 7}
    observed = {}

    def render(path, descriptor, source_identity, step, output, **options):
        observed.update(step=step, identity=source_identity, options=options)
        entered.set()
        assert release.wait(10)
        return {'record': {'step': step, 'identity': source_identity}, 'index': {}, 'errors': []}

    monkeypatch.setattr(module, 'render_snapshot', render)
    worker = PreviewWorker(timeout=3)
    temporary = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(temporary.name)
    worker.submit(temporary, {}, identity, 10, tmp_path, 2)
    try:
        assert entered.wait(5)
        assert worker.busy and worker.poll() is None
        assert directory.exists()
        with pytest.raises(RuntimeError, match='outstanding snapshot'):
            worker.submit(None, {}, {}, 11, tmp_path, 2)
    finally:
        release.set()
    result = worker.poll(wait=True)
    assert result['record'] == {'step': 10, 'identity': identity}
    assert 0 < observed['options'].pop('timeout') <= 3
    assert observed['options'] == {'publish_run_dir': tmp_path, 'keep': 2,
                                  'cancellation_event': worker._cancel}
    assert not worker.busy and not directory.exists()
    assert worker.poll(wait=True) is None


def test_failed_renderer_releases_snapshot_and_slot(tmp_path, monkeypatch):
    import hypergan.preview_worker as module
    def fail(*args, **kwargs):
        raise RuntimeError('render fault')
    monkeypatch.setattr(module, 'render_snapshot', fail)
    worker = PreviewWorker()
    temporary = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(temporary.name)
    worker.submit(temporary, {}, {}, 1, tmp_path, 1)
    with pytest.raises(RuntimeError, match='render fault'):
        worker.poll(wait=True)
    assert not worker.busy and not directory.exists()


@pytest.mark.heavy
@pytest.mark.parametrize('fail', [False, True])
def test_training_finishes_updates_while_preview_waits_and_keeps_source_step(tmp_path, monkeypatch, fail):
    import hypergan.preview_worker as module
    from hypergan.checkpoints import read_checkpoint
    from hypergan.config import write_default
    from hypergan.distributed_checkpoints import _digest
    from hypergan.training import train

    entered, release = Event(), Event()
    original = module.render_snapshot
    events = []

    def delayed(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError('Training did not advance while preview was pending')
        if fail:
            raise RuntimeError('Delayed preview failure')
        return original(*args, **kwargs)

    def observe(row):
        events.append(row)
        if row['event'] == 'train' and row['step'] == 5:
            assert entered.wait(5)
            release.set()

    config = write_default(tmp_path / 'config', device='cpu')
    train(config, tmp_path / 'plain', checkpoint_every=1)
    monkeypatch.setattr(module, 'render_snapshot', delayed)
    result = train(config, tmp_path / 'viewed', checkpoint_every=1, preview_every=1, on_event=observe)
    assert result['status'] == 'complete'
    assert result['skipped_previews_busy'] == 4
    assert [row['step'] for row in events if row['event'] == 'train'] == [1, 2, 3, 4, 5]
    previews = [row for row in events if row['event'] == 'preview']
    if fail:
        assert not previews
        assert result['observation_errors'][0]['step'] == 1
        errors = [row for row in events if row['event'] == 'observer_error']
        assert len(errors) == 1 and errors[0]['step'] == 1
    else:
        assert not result['observation_errors']
        assert len(previews) == 1 and previews[0]['step'] == previews[0]['preview']['step'] == 1
        assert Path(result['preview_path']).is_file()
    assert _digest(read_checkpoint(tmp_path / 'plain')[2]) == _digest(read_checkpoint(tmp_path / 'viewed')[2])


@pytest.mark.heavy
def test_fatal_update_cancels_hung_cpu_renderer_and_reaps_process(tmp_path, monkeypatch):
    import torch
    from hypergan.config import write_default
    from hypergan.training import ReferenceTrainer, train

    module = tmp_path / 'hanging_preview.py'
    marker = tmp_path / 'renderer.json'
    module.write_text('''import json, os, time, torch
from pathlib import Path
from hypergan.recipes import MLP
class Generator(MLP):
    def forward(self, x):
        if not self.training:
            Path(os.environ['PREVIEW_TEST_MARKER']).write_text(json.dumps({
                'pid': os.getpid(), 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'threads': torch.get_num_threads(), 'omp': os.environ.get('OMP_NUM_THREADS'),
                'nice': os.nice(0) if hasattr(os, 'nice') else None}))
            time.sleep(60)
        return super().forward(x)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PREVIEW_TEST_MARKER', str(marker))
    config = write_default(tmp_path / 'config', device='cpu')
    config.write_text(config.read_text().replace('factory = "hndl"', 'factory = "hanging_preview:Generator"', 1))
    original = ReferenceTrainer.update
    failed_at = []

    def update(trainer, *args, **kwargs):
        if trainer.step == 1:
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists(), 'Isolated renderer never reached its forward'
            failed_at.append(time.monotonic())
            raise ValueError('Injected fatal training failure')
        return original(trainer, *args, **kwargs)

    monkeypatch.setattr(ReferenceTrainer, 'update', update)
    with pytest.raises(ValueError, match='Injected fatal training failure'):
        train(config, tmp_path / 'run', preview_every=1, checkpoint_every=1)
    assert time.monotonic() - failed_at[0] < 10
    receipt = json.loads(marker.read_text())
    assert receipt['cuda'] == '' and receipt['threads'] == 1 and receipt['omp'] == '1'
    if hasattr(os, 'nice'):
        assert receipt['nice'] >= 10
    with pytest.raises(ProcessLookupError):
        os.kill(receipt['pid'], 0)
    assert not list((tmp_path / 'run').glob('.preview-*'))
    result = json.loads((tmp_path / 'run/manifest.json').read_text())
    assert result['status'] == 'failed' and result['last_durable_step'] == 1


def test_signal_arriving_during_terminal_preview_drain_cancels_promptly(tmp_path, monkeypatch):
    import hypergan.preview_worker as worker
    import hypergan.run_controller as controller
    from hypergan.config import write_default
    from hypergan.training import train

    class Stop:
        reason = None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    stop = Stop()
    monkeypatch.setattr(controller, 'GracefulStop', lambda: stop)
    entered = Event()
    timers = []

    def render(*args, cancellation_event, **kwargs):
        entered.set()
        assert cancellation_event.wait(5), 'Terminal drain ignored the stop request'
        raise RuntimeError('Preview cancelled')

    def observe(row):
        if row['event'] == 'train' and row['step'] == 5:
            assert entered.wait(5)
            timer = Timer(.05, lambda: setattr(stop, 'reason', 'SIGTERM'))
            timers.append(timer)
            timer.start()

    monkeypatch.setattr(worker, 'render_snapshot', render)
    config = write_default(tmp_path / 'config', device='cpu')
    started = time.monotonic()
    try:
        result = train(config, tmp_path / 'run', preview_every=1, on_event=observe)
    finally:
        for timer in timers:
            timer.cancel()
            timer.join()
    assert time.monotonic() - started < 3
    assert result['stop_reason'] == 'SIGTERM' and result['cancelled_previews'] == 1
    assert not result['observation_errors']


@pytest.mark.heavy
def test_blocked_snapshot_storage_allows_updates_and_preserves_complete_state(tmp_path, monkeypatch):
    import hypergan.preview_snapshot as snapshots
    from hypergan.checkpoints import read_checkpoint
    from hypergan.config import write_default
    from hypergan.distributed_checkpoints import _digest
    from hypergan.training import train, resume

    entered, release = Event(), Event()
    original_write = snapshots._BoundedWriter.write
    def blocked_write(writer, data):
        entered.set()
        assert release.wait(10), 'Training stalled behind snapshot storage'
        return original_write(writer, data)

    def observe(row):
        if row['event'] == 'train' and row['step'] == 3:
            assert entered.wait(5)
            release.set()

    config = write_default(tmp_path / 'config', device='cpu')
    train(config, tmp_path / 'plain', checkpoint_every=1)
    monkeypatch.setattr(snapshots._BoundedWriter, 'write', blocked_write)
    try:
        stopped = train(config, tmp_path / 'viewed', checkpoint_every=1, preview_every=1,
                        stop_after_steps=3, on_event=observe)
    finally:
        release.set()
    assert stopped['steps'] == 3 and stopped['status'] == 'stopped'
    assert not stopped['observation_errors']
    assert stopped['skipped_previews_busy'] == 2
    assert stopped['previews'][0]['step'] == 1
    done = resume(tmp_path / 'viewed', preview_every=0)
    assert done['status'] == 'complete'
    assert _digest(read_checkpoint(tmp_path / 'plain')[2]) == _digest(read_checkpoint(tmp_path / 'viewed')[2])
    assert not list((tmp_path / 'viewed').glob('.preview-*'))


@pytest.mark.parametrize('wait', [False, True])
def test_stalled_snapshot_storage_has_terminal_deadline_and_source_context(tmp_path, monkeypatch, wait):
    import hypergan.preview_snapshot as snapshots
    entered, release = Event(), Event()
    def stalled(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {}
    monkeypatch.setattr(snapshots, 'write_snapshot', stalled)
    worker = PreviewWorker(timeout=1)
    identity = {'run_id': 'run', 'attempt_id': 'attempt', 'sample_sequence': 7}
    worker.submit(None, None, identity, 3, tmp_path, 1, snapshot_state={})
    try:
        assert entered.wait(5)
        worker._deadline = time.monotonic() - 9
        with pytest.raises(TimeoutError, match='deadline and cleanup grace') as failure:
            worker.poll(wait=wait)
        assert failure.value.preview_context == {'step': 3, 'identity': identity}
        assert worker._cancel.is_set()
    finally:
        release.set()
        worker.abort()
    assert not list(tmp_path.glob('.preview-*'))


def test_signal_arriving_during_terminal_preview_drain_cancels_promptly(tmp_path, monkeypatch):
    import hypergan.preview_worker as worker
    import hypergan.run_controller as controller
    from hypergan.config import write_default
    from hypergan.training import train

    class Stop:
        reason = None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    stop = Stop()
    monkeypatch.setattr(controller, 'GracefulStop', lambda: stop)
    entered = Event()
    timers = []

    def render(*args, cancellation_event, **kwargs):
        entered.set()
        assert cancellation_event.wait(5), 'Terminal drain ignored the stop request'
        raise RuntimeError('Preview cancelled')

    def observe(row):
        if row['event'] == 'train' and row['step'] == 5:
            assert entered.wait(5)
            timer = Timer(.05, lambda: setattr(stop, 'reason', 'SIGTERM'))
            timers.append(timer)
            timer.start()

    monkeypatch.setattr(worker, 'render_snapshot', render)
    config = write_default(tmp_path / 'config', device='cpu')
    started = time.monotonic()
    try:
        result = train(config, tmp_path / 'run', preview_every=1, on_event=observe)
    finally:
        for timer in timers:
            timer.cancel()
            timer.join()
    assert time.monotonic() - started < 3
    assert result['stop_reason'] == 'SIGTERM' and result['cancelled_previews'] == 1
    assert not result['observation_errors']


def test_controller_terminal_poll_enforces_stalled_snapshot_storage_deadline(tmp_path, monkeypatch):
    import hypergan.preview_snapshot as snapshots
    from hypergan.config import write_default
    from hypergan.single_execution import SingleProcessExecution
    from hypergan.training import train

    entered = Event()
    instances = []
    original_preview = SingleProcessExecution.preview
    def preview(execution, *args, **kwargs):
        result = original_preview(execution, *args, **kwargs)
        instances.append(execution._previews)
        return result
    def stalled(state, path, *, cancellation_event, **kwargs):
        entered.set()
        assert cancellation_event.wait(5), 'Controller terminal poll ignored snapshot deadline'
        raise RuntimeError('Cancelled stalled snapshot storage')
    def observe(row):
        if row['event'] == 'train' and row['step'] == 5:
            assert entered.wait(5)
            instances[0]._deadline = time.monotonic() - 9
    monkeypatch.setattr(SingleProcessExecution, 'preview', preview)
    monkeypatch.setattr(snapshots, 'write_snapshot', stalled)
    config = write_default(tmp_path / 'config', device='cpu')
    started = time.monotonic()
    result = train(config, tmp_path / 'run', preview_every=1, on_event=observe)
    assert time.monotonic() - started < 3
    assert result['steps'] == 5 and result['status'] == 'complete'
    errors = result['observation_errors']
    assert len(errors) == 1 and errors[0]['source'] == 'preview' and errors[0]['step'] == 1
    assert 'deadline and cleanup grace' in errors[0]['error']
    assert not instances[0]._thread.is_alive()
    assert not list((tmp_path / 'run').glob('.preview-*'))
