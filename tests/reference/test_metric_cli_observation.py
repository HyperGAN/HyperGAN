"""The trusted CLI sink does not need numerical RNG isolation per event."""
import io
import inspect
import json

from hypergan.bounded_cli_output import TrainingOutput
from hypergan.config import write_default
from hypergan.training import train


def test_native_cli_progress_does_not_capture_numerical_rng(tmp_path, monkeypatch):
    import hypergan.single_execution as execution
    original = execution.capture_rng
    def capture():
        assert inspect.currentframe().f_back.f_code.co_name != 'observe'
        return original()
    monkeypatch.setattr(execution, 'capture_rng', capture)
    output = TrainingOutput(io.StringIO(), io.StringIO(), True, progress_every=1)
    config = write_default(tmp_path / 'config', device='cpu')
    result = train(config, tmp_path / 'run', on_event=output.progress)
    rows = [json.loads(line) for line in output.stdout.getvalue().splitlines()]
    assert [row['step'] for row in rows if row['event'] == 'train'] == list(range(1, 6))
    assert result['status'] == 'complete'


def test_user_callback_still_receives_rng_isolation(tmp_path, monkeypatch):
    import hypergan.single_execution as execution
    original = execution.capture_rng
    observed = []
    def capture():
        if inspect.currentframe().f_back.f_code.co_name == 'observe':
            observed.append(True)
        return original()
    monkeypatch.setattr(execution, 'capture_rng', capture)
    config = write_default(tmp_path / 'config', device='cpu')
    rows = []
    train(config, tmp_path / 'run', on_event=rows.append)
    assert len(observed) == len(rows) and len(rows) >= 5


def test_replicated_cli_progress_does_not_wait_for_health_roundtrip():
    from hypergan.replicated_execution import ReplicatedExecution
    output = TrainingOutput(io.StringIO(), io.StringIO(), True, progress_every=1)
    adapter = ReplicatedExecution.__new__(ReplicatedExecution)
    adapter.observer = output.progress
    adapter._closed = False
    class Service:
        def assert_healthy(self):
            raise AssertionError('CLI output must not wait for a worker health command')
    adapter.service = Service()
    adapter.observe(None, {'event': 'train', 'step': 1})
    assert json.loads(output.stdout.getvalue())['step'] == 1
