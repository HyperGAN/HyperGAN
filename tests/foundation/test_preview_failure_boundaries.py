"""Optional preview I/O must never hide failure of the numerical worker group."""
from pathlib import Path

import pytest

from hypergan.config import resolve_config
from hypergan.replicated_execution import ReplicatedExecution
from hypergan.run_controller import AttemptContext, FatalExecutionError


@pytest.mark.parametrize('failure', ['capture', 'renderer', 'publish', 'cleanup', 'interrupt'])
def test_cleanup_and_health_preserve_fatal_capture_or_interrupt(tmp_path, monkeypatch, failure):
    import hypergan.previews as previews
    import hypergan.replicated_execution as adapter
    import hypergan.snapshot_renderer as renderer
    execution = ReplicatedExecution(resolve_config({}),
        {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo'}})
    execution.configure_attempt(AttemptContext('run', 'attempt', 1, tmp_path, tmp_path / 'attempt'),
                                preview_every=1, on_event=None)

    class Service:
        aborted, checks = False, 0
        def command(self, *args):
            if failure == 'capture':
                raise RuntimeError('rank lost during capture')
            return {'results': [{'step': 0, 'ready': True, 'inference_available': False, 'snapshot': {}},
                                {'step': 0, 'ready': True, 'inference_available': False}]}
        def assert_healthy(self):
            self.checks += 1
            if self.aborted or self.checks > 1 and failure in ('publish', 'cleanup', 'interrupt'):
                raise RuntimeError('numerical worker group died')
        def _abort_preserving(self, error):
            self.aborted = True

    class Temporary:
        def __init__(self, **kwargs):
            self.name = str(tmp_path / 'temporary')
        def cleanup(self):
            raise OSError('temporary cleanup failed')

    def render(*args, **kwargs):
        if failure == 'renderer':
            raise RuntimeError('isolated renderer failed')
        if failure == 'interrupt':
            raise KeyboardInterrupt('user interrupted preview')
        return {}

    def publish(*args, **kwargs):
        if failure == 'publish':
            raise OSError('preview publication failed')
        return {}, {}, []

    execution.service = Service()
    monkeypatch.setattr(adapter.tempfile, 'TemporaryDirectory', Temporary)
    monkeypatch.setattr(renderer, 'render_snapshot', render)
    monkeypatch.setattr(previews, 'publish_preview_payload', publish)
    error = KeyboardInterrupt if failure == 'interrupt' else RuntimeError if failure == 'renderer' else FatalExecutionError
    message = {'capture': 'rank lost during capture', 'renderer': 'isolated renderer failed',
               'interrupt': 'user interrupted preview'}.get(failure, 'numerical worker group died')
    with pytest.raises(error, match=message):
        execution.preview(tmp_path, {'run_id': 'run', 'attempt_id': 'attempt'}, keep=3)
    assert execution._poisoned == (failure != 'renderer')
    assert execution.service.aborted == (failure != 'renderer')
