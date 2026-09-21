"""Actual pipe backpressure and process death, without numerical dependencies."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hypergan.bounded_cli_output import CLIProgress, MAX_LINE_BYTES, MAX_PENDING_LINES, training_output


def _env():
    env = os.environ.copy()
    # Tests also run against installed wheels outside the checkout.
    env['PYTHONPATH'] = os.pathsep.join(str(path) for path in sys.path if path)
    return env


def _alive(pid):
    if sys.platform == 'win32':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Linux container init may leave orphan zombies visible to kill(pid, 0).
    state = Path(f'/proc/{pid}/stat')
    if state.exists():
        return state.read_text().split()[2] != 'Z'
    return True


def _assert_dead(pids):
    deadline = time.monotonic() + 5
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(_alive(pid) for pid in pids)


def test_json_format_and_native_diagnostics():
    code = '''
from hypergan.bounded_cli_output import training_output
import os,sys,subprocess
with training_output(progress_json=True, progress_every=1) as output:
 output.progress({'event':'train','step':1})
 print('warning: fixture', file=sys.stderr)
 os.write(2,b'native fixture\\n')
 subprocess.run([sys.executable,'-c',"import os; os.write(2,b'inherited fixture\\\\n')"],check=True)
 output.result({'step':1})
'''
    result = subprocess.run([sys.executable, '-c', code], env=_env(), capture_output=True, timeout=10)
    assert result.returncode == 0
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {'event': 'train', 'step': 1}, {'event': 'result', 'manifest': {'step': 1}}]
    assert b'warning: fixture\n' in result.stderr
    assert b'native fixture\n' in result.stderr
    assert b'inherited fixture\n' in result.stderr


@pytest.mark.parametrize('destination', ['unread', 'closed', 'slow'])
def test_consumer_cannot_hold_terminal_completion(tmp_path, destination):
    receipt = tmp_path / 'receipt.json'
    code = '''
from hypergan.bounded_cli_output import training_output
import json,os,sys
from pathlib import Path
with training_output(progress_json=True, progress_every=1) as output:
 pids=[output.stdout.process.pid,output.stderr.process.pid]
 for step in range(2000):
  output.progress({'event':'train','step':step,'payload':'x'*32000})
  print('diagnostic'*3000,file=sys.stderr)
 # Native worker/library writes are drained independently too.
 for _ in range(100):
  os.write(1,b'native stdout\\n'*3000)
  os.write(2,b'native stderr\\n'*3000)
 output.result({'step':2000})
Path(sys.argv[1]).write_text(json.dumps(pids))
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(receipt)], env=_env(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if destination == 'closed':
        process.stdout.close()
        process.stderr.close()
    if destination == 'slow':
        import threading
        def sip(stream):
            try:
                while stream.read(1):
                    time.sleep(0.005)
            except ValueError:
                pass
        # Threads only touch unbuffered raw reads, never interpreter text locks.
        readers = [threading.Thread(target=sip, args=(stream,), daemon=True)
                   for stream in (process.stdout, process.stderr)]
        for reader in readers:
            reader.start()
    try:
        assert process.wait(timeout=15) == 0
        _assert_dead(json.loads(receipt.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdout, process.stderr):
            if not stream.closed:
                stream.close()


def test_parent_death_reaps_blocked_drains(tmp_path):
    receipt = tmp_path / 'pids.json'
    code = '''
from hypergan.bounded_cli_output import training_output
import json,sys,time
from pathlib import Path
with training_output(progress_json=True, progress_every=1) as output:
 Path(sys.argv[1]).write_text(json.dumps([output.stdout.process.pid,output.stderr.process.pid]))
 for step in range(200):
  output.progress({'event':'train','step':step,'payload':'x'*32000})
 time.sleep(60)
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(receipt)], env=_env(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pids = json.loads(receipt.read_text())
        time.sleep(0.2)
        process.kill()
        process.wait(timeout=5)
        _assert_dead(pids)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def test_memory_capture_oversize_and_exact_sink(monkeypatch):
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    with training_output(progress_json=True, progress_every=1) as output:
        assert type(output.progress) is CLIProgress
        output.stdout.write('x' * (MAX_LINE_BYTES + 1) + '\n')
        output.stdout.write('é' * MAX_LINE_BYTES + '\n')
        output.progress({'event': 'train', 'step': 2})
        assert output.stdout.dropped_lines == 2
        assert output.stdout.pending.maxsize == MAX_PENDING_LINES
    assert json.loads(stdout.getvalue()) == {'event': 'train', 'step': 2}
    assert stderr.getvalue() == ''
    assert sys.stdout is stdout


def test_exception_restores_streams(capsys):
    originals = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError, match='fixture'):
        with training_output():
            print('error: fixture', file=sys.stderr)
            raise RuntimeError('fixture')
    assert (sys.stdout, sys.stderr) == originals
    assert capsys.readouterr().err == 'error: fixture\n'


def test_large_normal_result_remains_one_complete_json_record():
    code = """
from hypergan.bounded_cli_output import training_output
with training_output() as output:
 output.result({str(index): 'manifest field' * 20 for index in range(150)})
"""
    result = subprocess.run([sys.executable, '-c', code], env=_env(), capture_output=True, timeout=10)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {str(index): 'manifest field' * 20 for index in range(150)}
    assert len(result.stdout.splitlines()) == 1


def test_oversized_result_has_explicit_manifest_fallback(capsys, tmp_path):
    with training_output(progress_json=True, progress_every=1) as output:
        output.result({'payload': 'x' * MAX_LINE_BYTES}, run_dir=tmp_path)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        'event': 'output_omitted', 'reason': 'result_exceeds_output_limit',
        'read': str(tmp_path / 'manifest.json')}
    assert 'read the durable run manifest.json' in captured.err


_BUFFERED_CODE = '''
from hypergan.bounded_cli_output import training_output,_native_standard_streams
import ctypes,json,os,sys,time
from pathlib import Path
runtime,streams = _native_standard_streams()[0]
runtime.fputs.argtypes = [ctypes.c_char_p,ctypes.c_void_p]
runtime.setvbuf.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
for stream in streams:
 assert runtime.setvbuf(stream,None,0,4096) == 0
sys.__stdout__.write('cached-before')
with training_output(progress_json=True, progress_every=1) as output:
 pids=[output.stdout.process.pid,output.stderr.process.pid]
 if sys.argv[2] == 'unread':
  for step in range(100):
   output.progress({'event':'train','step':step,'data':'x'*32000})
   print('diagnostic'*3000,file=sys.stderr)
  time.sleep(0.3)
 sys.__stdout__.write('cached-python-stdout')
 sys.__stderr__.write('cached-python-stderr')
 runtime.fputs(b'cached-native-stdout',streams[0])
 runtime.fputs(b'cached-native-stderr',streams[1])
 output.result({'step':100})
Path(sys.argv[1]).write_text(json.dumps(pids))
'''


def test_cached_python_and_native_stdio_cannot_hang_exit(tmp_path):
    receipt = tmp_path / 'pids.json'
    process = subprocess.Popen([sys.executable, '-c', _BUFFERED_CODE, str(receipt), 'unread'],
                               env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.wait(timeout=12) == 0
        _assert_dead(json.loads(receipt.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def test_cached_python_and_native_stdio_are_forwarded(tmp_path):
    result = subprocess.run([sys.executable, '-c', _BUFFERED_CODE, str(tmp_path / 'pids'), 'healthy'],
                            env=_env(), capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert b'cached-before' in result.stdout
    assert b'cached-python-stdout' in result.stdout
    assert b'cached-native-stdout' in result.stdout
    assert b'cached-python-stderr' in result.stderr
    assert b'cached-native-stderr' in result.stderr


def test_flush_failure_still_restores_and_reaps(monkeypatch, capfd):
    from hypergan.bounded_cli_output import _NativeStreams
    originals = sys.stdout, sys.stderr
    def fail(_):
        raise RuntimeError('native flush fixture')
    monkeypatch.setattr(_NativeStreams, 'flush', fail)
    with pytest.raises(RuntimeError, match='native flush fixture'):
        with training_output() as output:
            pids = [output.stdout.process.pid, output.stderr.process.pid]
    assert (sys.stdout, sys.stderr) == originals
    _assert_dead(pids)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal process-group signals')
@pytest.mark.parametrize('signum', [2, 15])
def test_terminal_group_signal_preserves_final_output(tmp_path, signum):
    import signal
    ready = tmp_path / 'ready'
    code = '''
import signal,time,sys
from pathlib import Path
from hypergan.bounded_cli_output import training_output
stopped=[]
signal.signal(signal.SIGINT, lambda *args: stopped.append(True))
signal.signal(signal.SIGTERM, lambda *args: stopped.append(True))
with training_output(progress_json=True) as output:
 Path(sys.argv[1]).write_text('ready')
 while not stopped: time.sleep(.01)
 output.result({'status':'stopped','steps':7})
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(ready)], env=_env(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        os.killpg(process.pid, signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert json.loads(stdout) == {'event': 'result', 'manifest': {'status': 'stopped', 'steps': 7}}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize('seconds,expected', [
    (0, '0s'), (0.9, '0s'), (45, '45s'), (59.9, '59s'), (60, '1m 00s'),
    (725, '12m 05s'), (3600, '1h 00m'), (5025, '1h 23m'), (360000, '100h 00m'),
])
def test_training_duration_reads_as_a_bounded_human_value(seconds, expected):
    from hypergan.bounded_cli_output import human_duration
    assert human_duration(seconds) == expected


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), None, '60'])
def test_unusable_duration_prints_nothing_rather_than_a_wrong_one(value):
    from hypergan.bounded_cli_output import human_duration
    assert human_duration(value) is None


def test_progress_line_reports_throughput_training_time_and_samples(capsys):
    with training_output(progress_every=1) as output:
        output.progress({'event': 'train', 'step': 7, 'metrics': {'loss/d_total': 1.5, 'loss/g_total': 2.5},
                         'steps_per_second': 123.44, 'training_seconds': 5025.4, 'samples_seen': 1792})
        # A slow run keeps three significant digits instead of reporting 0.0.
        output.progress({'event': 'train', 'step': 8, 'metrics': {},
                         'steps_per_second': .0421, 'training_seconds': 45., 'samples_seen': 2048})
        # An update whose throughput is not measurable yet omits only that field.
        output.progress({'event': 'train', 'step': 9, 'metrics': {}, 'training_seconds': 0., 'samples_seen': 2304})
        # An event without progress fields keeps the original one-line shape.
        output.progress({'event': 'train', 'step': 10})
    assert capsys.readouterr().err.splitlines() == [
        'step 7: D=1.5 G=2.5 | 123.4 steps/s | 1h 23m | 1,792 samples',
        'step 8 | 0.0421 steps/s | 45s | 2,048 samples',
        'step 9 | 0s | 2,304 samples',
        'step 10',
    ]


def test_json_progress_rows_carry_the_same_training_metrics(capsys):
    with training_output(progress_json=True, progress_every=1) as output:
        output.progress({'event': 'train', 'step': 3, 'steps_per_second': 12.5,
                         'training_seconds': 1.25, 'samples_seen': 96})
    row = json.loads(capsys.readouterr().out)
    assert row['steps_per_second'] == 12.5 and row['training_seconds'] == 1.25 and row['samples_seen'] == 96
