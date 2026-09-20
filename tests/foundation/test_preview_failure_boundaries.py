"""Optional preview I/O must never hide failure of the numerical worker group."""
from pathlib import Path

import pytest

from hypergan.config import resolve_config
from hypergan.replicated_execution import ReplicatedExecution
from hypergan.run_controller import AttemptContext, FatalExecutionError


@pytest.mark.parametrize('failure', ['capture', 'renderer', 'publish', 'cleanup', 'interrupt',
                                     'system_exit', 'fatal_renderer'])
def test_cleanup_and_health_preserve_fatal_capture_or_interrupt(tmp_path, monkeypatch, failure):
    import hypergan.previews as previews
    import hypergan.replicated_execution as adapter
    import hypergan.preview_worker as renderer
    execution = ReplicatedExecution(resolve_config({}),
        {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo'}})
    execution.configure_attempt(AttemptContext('run', 'attempt', 1, tmp_path, tmp_path / 'attempt'),
                                preview_every=1, on_event=None)

    class Service:
        aborted, dead = False, False
        def command(self, *args):
            if failure == 'capture':
                raise RuntimeError('rank lost during capture')
            return {'results': [{'step': 0, 'ready': True, 'inference_available': False, 'snapshot': {}},
                                {'step': 0, 'ready': True, 'inference_available': False}]}
        def assert_healthy(self):
            if self.aborted or self.dead:
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
        if failure == 'system_exit':
            raise SystemExit('user exited preview')
        if failure == 'fatal_renderer':
            raise FatalExecutionError('fatal preview failure')
        publish()
        return {}

    def publish(*args, **kwargs):
        if failure == 'publish':
            raise OSError('preview publication failed')
        return {}, {}, []

    execution.service = Service()
    monkeypatch.setattr(adapter.tempfile, 'TemporaryDirectory', Temporary)
    monkeypatch.setattr(renderer, 'render_snapshot', render)
    monkeypatch.setattr(previews, 'publish_preview_payload', publish)
    error = (KeyboardInterrupt if failure == 'interrupt' else SystemExit if failure == 'system_exit'
             else RuntimeError if failure == 'renderer' else FatalExecutionError)
    message = {'capture': 'rank lost during capture', 'renderer': 'isolated renderer failed',
               'interrupt': 'user interrupted preview', 'system_exit': 'user exited preview',
               'fatal_renderer': 'fatal preview failure'}.get(failure, 'numerical worker group died')
    identity={'run_id': 'run', 'attempt_id': 'attempt'}
    if failure=='capture':
        with pytest.raises(error, match=message):
            execution.preview(tmp_path, identity, keep=3)
    else:
        # Capture/admission succeeds before the optional renderer completes.
        execution.preview(tmp_path, identity, keep=3)
        execution.service.dead = failure!='renderer'
        with pytest.raises(error, match=message) as caught:
            execution.poll_preview(wait=True)
        if failure in ('renderer', 'interrupt', 'system_exit', 'fatal_renderer'):
            assert caught.value.preview_context == {'step':0,'identity':identity}
        execution._previews._thread.join(1)
        assert not execution._previews._thread.is_alive()
        assert not execution.preview_busy
    assert execution._poisoned == (failure != 'renderer')
    assert execution.service.aborted == (failure != 'renderer')
