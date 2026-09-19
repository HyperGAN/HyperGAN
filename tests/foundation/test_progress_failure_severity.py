"""Progress cleanup cannot turn a primary interrupt into an optional failure."""
import pytest

from hypergan.bounded_observer import ObserverError
from hypergan.config import resolve_config
from hypergan.replicated_execution import ReplicatedExecution
from hypergan.run_controller import FatalExecutionError


@pytest.mark.parametrize('primary,healthy,expected', [
    (KeyboardInterrupt('user interrupt'), False, KeyboardInterrupt),
    (SystemExit('user exit'), False, SystemExit),
    (FatalExecutionError('primary fatal error'), False, FatalExecutionError),
    (ObserverError('optional callback failed'), True, ObserverError),
    (ObserverError('optional callback failed'), False, FatalExecutionError),
])
def test_progress_health_failure_preserves_primary_severity(primary, healthy, expected):
    execution = ReplicatedExecution(resolve_config({}),
        {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo'}})

    class Observer:
        disabled = False
        def deliver(self, event):
            raise primary

    class Service:
        aborted = False
        def assert_healthy(self):
            if not healthy:
                raise RuntimeError('training rank died during callback')
        def _abort_preserving(self, error):
            self.aborted = True

    execution.observer, execution.service = Observer(), Service()
    with pytest.raises(expected) as raised:
        execution.observe(None, {})
    if healthy or isinstance(primary, (KeyboardInterrupt, SystemExit, FatalExecutionError)):
        assert raised.value is primary
    else:
        assert 'training rank died during callback' in str(raised.value)
    assert execution._poisoned == (not healthy)
    assert execution.service.aborted == (not healthy)
