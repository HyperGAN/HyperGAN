"""The service parent/control format requires no numerical dependencies."""
import socket
import subprocess
import sys

import pytest

from hypergan.cpu_worker_service import CPUWorkerService, MAX_FRAME_BYTES, _Channel, _json, _validate


@pytest.mark.parametrize('kwargs', [{'world_size': True}, {'world_size': 1}, {'world_size': 65},
                                   {'run_id': '../bad'}, {'attempt_id': ''}, {'command_timeout': False},
                                   {'total_timeout': float('inf')}, {'startup_timeout': 0}])
def test_invalid_service_controls(kwargs):
    values = dict(run_id='run-one', attempt_id='attempt-one')
    values.update(kwargs)
    with pytest.raises(ValueError):
        CPUWorkerService(str, str, **values)


def test_identity_and_timeout_policy_are_not_mutable_through_public_fields():
    service = CPUWorkerService(str, str, run_id='run', attempt_id='attempt')
    service.limits['total_timeout'] = 0
    assert service.limits['total_timeout'] == 300
    with pytest.raises(AttributeError):
        service.identity = ('changed', 'attempt')
    assert service.sequence == 0 and service.next_sequence == 1
    service.close()
    with pytest.raises(RuntimeError):
        service.start()


@pytest.mark.parametrize('value', [{1: 'bad'}, ('tuple',), float('nan'), b'bytes', {'large':'x'*MAX_FRAME_BYTES}])
def test_json_wire_contract(value):
    with pytest.raises(ValueError):
        _json(value)


def test_partial_frame_is_not_an_early_command_and_bad_length_fails():
    left, right = socket.socketpair()
    channel = _Channel(left)
    try:
        right.sendall(b'\0\0\0\x02{')
        assert channel.pump() == []
        right.sendall(b'}')
        assert channel.pump() == [{}]
        right.sendall(b'\xff\xff\xff\xff')
        with pytest.raises(ValueError, match='frame length'):
            channel.pump()
    finally:
        channel.close()
        right.close()


@pytest.mark.parametrize('change', [{'run_id':'other'}, {'attempt_id':'old'}, {'sequence':True},
                                   {'sequence':0}, {'sequence':2}, {'operation':'other'}])
def test_identity_and_command_sequence_fence(change):
    row = dict(run_id='run', attempt_id='attempt', sequence=1, operation='save')
    row.update(change)
    with pytest.raises(ValueError, match='Stale or invalid'):
        _validate(row, ('run','attempt'), 1, 'save')


@pytest.mark.parametrize('operation,fields', [
    ('__start__', {'kind':'ready', 'worker_pids':[1, True]}),
    ('__start__', {'kind':'ready', 'worker_pids':[1, 1]}),
    ('__health__', {'kind':'result', 'results':[None, None]}),
    ('__shutdown__', {'kind':'closed', 'unexpected':True}),
    ('save', {'kind':'result', 'results':[None]}),
])
def test_malformed_matching_fence_replies_are_rejected(operation, fields):
    service = CPUWorkerService(str, str, run_id='run', attempt_id='attempt')
    row = dict(fields, run_id='run', attempt_id='attempt', sequence=0, operation=operation)
    class Channel:
        def pump(self):
            return [row]
    service._channel = Channel()
    with pytest.raises(RuntimeError, match='result schema or rank count'):
        service._reply(operation, 1)


def test_service_import_is_torch_free(tmp_path):
    script = '''
import importlib.abc,sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','particlegan','PIL'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0,NoRuntime())
from hypergan.cpu_worker_service import CPUWorkerService
CPUWorkerService(str,str,run_id='run',attempt_id='attempt').close()
'''
    subprocess.run([sys.executable,*(['-I'] if sys.flags.isolated else []),'-c',script],cwd=tmp_path,check=True,timeout=10)
