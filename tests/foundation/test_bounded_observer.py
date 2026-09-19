"""Progress isolation needs no optional numerical dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hypergan.bounded_observer import BoundedObserver, ObserverError, callback_reference


def record(event):
    Path(event['path']).write_text(json.dumps({'event': event, 'pid': os.getpid(),
                                             'torch_imported': 'torch' in sys.modules}))
    event['mutated'] = True
    return object()  # Results are deliberately ignored, not serialized.


def fail(event):
    raise RuntimeError('deliberate callback failure')


def hang(event):
    Path(event['path']).write_text(str(os.getpid()))
    time.sleep(120)


def no_op(event):
    pass


def assert_gone(pids):
    if sys.platform != 'linux':
        return  # Process joins are also asserted through service completion.
    assert all(not Path(f'/proc/{pid}').exists() for pid in pids)


def test_success_snapshot_no_torch_and_reaped(tmp_path):
    observer = BoundedObserver(record, timeout=10, run_id='run', attempt_id='attempt')
    assert observer.broker_pid is None
    event = {'path': str(tmp_path / 'event.json'), 'step': 7}
    assert observer.deliver(event) is True
    value = json.loads(Path(event['path']).read_text())
    assert value['event'] == event
    assert value['pid'] in observer.worker_pids
    assert not value['torch_imported']
    assert 'mutated' not in event
    assert_gone([observer.broker_pid, *observer.worker_pids])
    first = observer.worker_pids[:]
    assert observer.deliver(event) is True
    assert observer.worker_pids != first
    assert_gone([observer.broker_pid, *observer.worker_pids])
    observer.close()
    with pytest.raises(RuntimeError, match='closed'):
        observer.deliver(event)


def test_callback_error_disables_once_and_reaps():
    observer = BoundedObserver(fail, timeout=10)
    with pytest.raises(ObserverError, match='deliberate callback failure.*',):
        observer.deliver({})
    assert observer.disabled
    pids = [observer.broker_pid, *observer.worker_pids]
    assert_gone(pids)
    assert observer.deliver({}) is False
    assert pids == [observer.broker_pid, *observer.worker_pids]


def test_hang_bounded_and_reaped(tmp_path):
    observer = BoundedObserver(hang, timeout=2)
    started = time.monotonic()
    with pytest.raises(ObserverError, match='deadline'):
        observer.deliver({'path': str(tmp_path / 'pid')})
    assert time.monotonic() - started < 10
    assert observer.disabled
    assert_gone([observer.broker_pid, *observer.worker_pids])
    assert int((tmp_path / 'pid').read_text()) in observer.worker_pids


@pytest.mark.parametrize('event', [[], {'x': float('nan')}, {'x': object()},
                                    {'x': 'a' * 65536}, {1: 'value'}])
def test_invalid_event_rejected_before_process(event):
    observer = BoundedObserver(no_op)
    with pytest.raises(ValueError):
        observer.deliver(event)
    assert observer.broker_pid is None
    assert not observer.disabled


def test_exact_event_size_limit(tmp_path):
    # JSON {'x': '...'} has eight framing bytes; the control envelope remains
    # small because the validated event is part of the bounded bootstrap args.
    observer = BoundedObserver(no_op, timeout=10)
    assert observer.deliver({'x': 'a' * (65536 - 8)})
    with pytest.raises(ValueError, match='65536'):
        observer.deliver({'x': 'a' * (65536 - 7)})


def test_static_callback_validation_and_single_outstanding():
    for callback in (lambda event: None, object(), print):
        with pytest.raises(ValueError, match='module-level'):
            BoundedObserver(callback)
    def closure(event):
        pass
    with pytest.raises(ValueError, match='module-level'):
        BoundedObserver(closure)
    assert callback_reference(no_op).endswith(':no_op')
    observer = BoundedObserver(no_op)
    observer._delivery_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='one progress'):
            observer.deliver({})
        with pytest.raises(RuntimeError, match='during progress'):
            observer.close()
    finally:
        observer._delivery_lock.release()


@pytest.mark.parametrize('timeout', [0, -1, True, float('inf'), '5'])
def test_invalid_deadline_before_process(timeout):
    with pytest.raises(ValueError, match='finite positive'):
        BoundedObserver(no_op, timeout=timeout)


def test_module_import_is_torch_free():
    code = "import sys; import hypergan.bounded_observer; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', code], check=True, timeout=10)


def test_blocked_output_is_bounded(tmp_path):
    driver = tmp_path / 'blocked_output.py'
    driver.write_text('''
import os
from pathlib import Path
from hypergan.bounded_observer import BoundedObserver, ObserverError

def noisy(event):
    Path(event['pid']).write_text(str(os.getpid()))
    while True:
        os.write(2, b'x' * 65536)

if __name__ == '__main__':
    observer = BoundedObserver(noisy, timeout=2)
    try:
        observer.deliver({'pid': __file__ + '.pid'})
    except ObserverError as error:
        assert 'deadline' in str(error)
        Path(__file__ + '.done').write_text('reaped')
    else:
        raise AssertionError('Expected blocked output to exceed deadline')
''')
    process = subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []), str(driver)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Deliberately do not drain the worker's inherited stderr pipe.
        assert process.wait(timeout=12) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
    assert Path(str(driver) + '.done').read_text() == 'reaped'
    assert_gone([int(Path(str(driver) + '.pid').read_text())])


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux parent-death fixture uses prctl subreaper and native libc sleep')
def test_parent_death_during_native_callback_reaps_child(tmp_path):
    driver = tmp_path / 'parent_death.py'
    driver.write_text('''
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from hypergan.bounded_observer import BoundedObserver

def native_hang(event):
    Path(event['marker']).write_text(json.dumps({'worker': os.getpid(), 'broker': os.getppid()}))
    ctypes.PyDLL(None).sleep(120)  # Holds Python's GIL in the observer process.

if __name__ == '__main__':
    marker = Path(__file__ + '.pids')
    if len(sys.argv) > 1:
        BoundedObserver(native_hang, timeout=60).deliver({'marker': str(marker)})
    else:
        assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
        coordinator = subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []), __file__, 'coordinator'])
        pids = {}
        try:
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and not marker.exists():
                assert coordinator.poll() is None
                time.sleep(.02)
            pids = json.loads(marker.read_text())
            coordinator.kill()
            coordinator.wait(timeout=5)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and Path('/proc/' + str(pids['worker'])).exists():
                time.sleep(.02)
            assert not Path('/proc/' + str(pids['worker'])).exists(), 'Observer child, including zombie, leaked'
            while time.monotonic() < deadline:
                try:
                    result, status = os.waitpid(pids['broker'], os.WNOHANG)
                except ChildProcessError:
                    result = pids['broker']
                if result:
                    break
                time.sleep(.02)
            else:
                raise AssertionError('Broker did not exit after reaping callback child')
            Path(__file__ + '.done').write_text('reaped')
        finally:
            if coordinator.poll() is None:
                coordinator.kill()
            coordinator.wait(timeout=5)
            for pid in pids.values():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for pid in pids.values():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        result, status = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if result:
                        break
                    time.sleep(.02)
''')
    subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), str(driver)],
                   check=True, timeout=40, capture_output=True, text=True)
    assert Path(str(driver) + '.done').read_text() == 'reaped'
