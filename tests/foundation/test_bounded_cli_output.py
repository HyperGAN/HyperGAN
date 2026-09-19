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
with training_output(progress_json=True) as output:
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
with training_output(progress_json=True) as output:
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
with training_output(progress_json=True) as output:
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
    with training_output(progress_json=True) as output:
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
    with training_output(progress_json=True) as output:
        output.result({'payload': 'x' * MAX_LINE_BYTES}, run_dir=tmp_path)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        'event': 'output_omitted', 'reason': 'result_exceeds_output_limit',
        'read': str(tmp_path / 'manifest.json')}
    assert 'read the durable run manifest.json' in captured.err
