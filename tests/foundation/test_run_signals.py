"""Signal escalation has a wall-clock bound independent of update progress."""
import os
import signal
import subprocess
import sys

import pytest

from hypergan.run_signals import GracefulStop


def test_signal_handlers_restore_host_policy():
    before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    with GracefulStop() as stop:
        assert stop.reason is None
    assert all(signal.getsignal(number) == handler for number, handler in before.items())


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signal delivery; Windows process termination is forced')
@pytest.mark.parametrize('mode', ['deadline', 'second'])
def test_blocked_shutdown_and_second_signal_force_exit(mode):
    script = '''
import os, signal, time
from hypergan.run_signals import GracefulStop
with GracefulStop(timeout=.15):
    os.kill(os.getpid(), signal.SIGTERM)
    if MODE == 'second':
        os.kill(os.getpid(), signal.SIGINT)
    time.sleep(60)
'''.replace('MODE', repr(mode))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 128 + (signal.SIGTERM if mode == 'deadline' else signal.SIGINT), result.stderr
