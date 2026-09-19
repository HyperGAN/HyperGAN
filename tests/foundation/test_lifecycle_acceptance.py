"""Independent resource ordering without numerical execution dependencies."""
import json

import pytest

from hypergan.config import write_default
from hypergan.run_state import run_lock


class ProbeExecution:
    """Resource-owning execution with no numerical trainer or controller internals."""

    def __init__(self, config, *, root, trace, fail=None, fail_shutdown=False):
        self.root = root
        self.trace = trace
        self.fail = fail
        self.fail_shutdown = fail_shutdown
        self.step = 0
        self.closed = False

    @staticmethod
    def environment():
        return {'runtime': {'kind': 'independent-test'}, 'source': {}}

    def start(self):
        from hypergan.run_controller import ExecutionInfo
        self.trace.append(('start', self.step))
        if self.fail == 'start':
            raise RuntimeError('primary start failure')
        return ExecutionInfo(step=self.step, data_identity={'fixture': 'probe'},
                             recovery_reasons=[], checkpoint_metadata={})

    def update(self):
        from hypergan.run_controller import CompletedUpdate
        self.trace.append(('update', self.step))
        if self.fail == 'update':
            raise RuntimeError('primary update failure after partial work')
        if self.fail == 'interrupt':
            raise KeyboardInterrupt('primary interruption after partial work')
        self.step += 1
        return CompletedUpdate(step=self.step, metrics={'loss': 0.25})

    def checkpoint(self, run_dir, metadata):
        self.trace.append(('checkpoint', self.step))
        if self.fail == 'checkpoint' and self.step:
            raise OSError('primary checkpoint failure')
        target = run_dir / 'probe-checkpoints' / str(self.step)
        target.mkdir(parents=True)
        (target / 'manifest.json').write_text(json.dumps(dict(metadata, step=self.step)), encoding='utf-8')
        return target

    @property
    def inference_available(self):
        return self.step > 0

    def inference(self, bundle_dir, identity):
        from hypergan.run_controller import ArtifactResult
        self.trace.append(('inference', self.step))
        if self.fail == 'inference':
            raise RuntimeError('primary inference failure')
        model, sample = bundle_dir / 'model.pt', bundle_dir / 'sample.json'
        model.write_bytes(b'probe')
        sample.write_text('{}', encoding='utf-8')
        return ArtifactResult(bundle_path=model, sample_path=sample)

    def observe(self, callback, event):
        callback(event)

    def shutdown(self):
        if self.closed:
            return
        # Resources must still be fenced by the run lock, and observers must not
        # already have seen a successful or failed terminal status.
        assert json.loads((self.root / 'manifest.json').read_text())['status'] in ('initializing', 'running')
        with pytest.raises(RuntimeError):
            with run_lock(self.root):
                pytest.fail('run lock released before execution cleanup')
        self.trace.append(('shutdown', self.step))
        self.closed = True
        if self.fail == 'shutdown' or self.fail_shutdown:
            raise RuntimeError('secondary shutdown failure')


def _run_probe(tmp_path, **options):
    from hypergan.run_controller import run_train
    config = write_default(tmp_path / 'config')
    root = tmp_path / 'run'
    trace, instances = [], []

    def factory(config):
        execution = ProbeExecution(config, root=root, trace=trace, **options)
        instances.append(execution)
        return execution

    factory.environment = ProbeExecution.environment

    def observer(event):
        trace.append(('event:' + event['event'], event['step']))
        if event['event'] in ('complete', 'stopped', 'failed', 'interrupted'):
            assert instances[0].closed, 'terminal event preceded execution cleanup'
            assert json.loads((root / 'manifest.json').read_text())['status'] == event['event']

    return root, trace, instances, lambda **controls: run_train(
        config, root, execution_factory=factory, on_event=observer, **controls)


@pytest.mark.parametrize('stopped', [False, True])
def test_controller_requires_cleanup_before_terminal_success(tmp_path, stopped):
    root, trace, executions, run = _run_probe(tmp_path)
    result = run(stop_after_steps=2 if stopped else None)
    status, step = ('stopped', 2) if stopped else ('complete', 5)
    assert result['status'] == status and result['steps'] == result['last_durable_step'] == step
    assert trace.index(('checkpoint', step)) < trace.index(('inference', step))
    assert trace.index(('inference', step)) < trace.index(('shutdown', step)) < trace.index(('event:' + status, step))
    assert executions[0].closed
    with run_lock(root):
        pass  # Ownership is released after the whole attempt has terminated.


@pytest.mark.parametrize('failure', ['start', 'update', 'interrupt', 'checkpoint', 'inference', 'shutdown'])
def test_controller_failure_cleans_up_without_half_update_checkpoint(tmp_path, failure):
    root, trace, executions, run = _run_probe(tmp_path, fail=failure)
    expected = KeyboardInterrupt if failure == 'interrupt' else OSError if failure == 'checkpoint' else RuntimeError
    with pytest.raises(expected):
        run(checkpoint_every=1)
    manifest = json.loads((root / 'manifest.json').read_text())
    status = 'interrupted' if failure == 'interrupt' else 'failed'
    assert manifest['status'] == status and executions[0].closed
    shutdown_index = next(i for i, row in enumerate(trace) if row[0] == 'shutdown')
    assert all(row[0] not in ('update', 'checkpoint', 'inference') for row in trace[shutdown_index + 1:])
    assert trace[-1][0] == 'event:' + status
    assert not any(row[0] in ('event:complete', 'event:stopped') for row in trace)
    if failure in ('update', 'interrupt'):
        assert [row for row in trace if row[0] == 'checkpoint'] == [('checkpoint', 0)]
        assert manifest['steps'] == manifest['last_durable_step'] == 0
    if failure == 'checkpoint':
        assert manifest['steps'] == 1 and manifest['last_durable_step'] == 0
        assert manifest['possible_lost_steps'] == 1
    with run_lock(root):
        pass


def test_controller_preserves_primary_failure_when_cleanup_also_fails(tmp_path):
    root, trace, executions, run = _run_probe(tmp_path, fail='update', fail_shutdown=True)
    with pytest.raises(RuntimeError, match='primary update failure'):
        run()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert 'primary update failure' in manifest['error']
    assert executions[0].closed and trace[-1][0] == 'event:failed'
    assert [row for row in trace if row[0] == 'checkpoint'] == [('checkpoint', 0)]


