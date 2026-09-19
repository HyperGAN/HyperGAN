"""Parent publication protocol tests using opaque, prevalidated rank bytes.

Actual tensor preparation and fresh-group restore live in the CPU reference
suite. The parent deliberately verifies hashes without importing tensor loaders.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import hypergan.distributed_commit as commit
from hypergan.run_state import atomic_json, run_lock


def _fixture(run, *, sequence=1, authority=None):
    run.mkdir(exist_ok=True)
    identity = {'config': {'training': {'steps': 3}}, 'topology': {'world_size': 2}}
    authority = authority or commit.CheckpointCommitAuthority(run, run_id='run-one', attempt_id='attempt-one', identity=identity)
    nonce = f'{sequence:012x}'
    staging = commit.preparation_directory(run, 'attempt-one', authority.controller_id, sequence, nonce, create=True)
    records = []
    for rank in range(2):
        payload = f'opaque rank {rank}'.encode()
        name = f'rank-{rank:05d}.pt'
        (staging / name).write_bytes(payload)
        records.append({'rank': rank, 'file': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    info = {'schema_version': 1, 'kind': commit.KIND, 'run_id': 'run-one', 'attempt_id': 'attempt-one',
            'step': 2, 'identity': identity, 'replicated_sha256': 'a' * 64, 'ranks': records,
            'next_sample_sequence': 3, 'request_ids': ['request-one']}
    atomic_json(staging / 'manifest.json', info)
    receipt = commit.make_prepared_receipt(run, staging, info, controller_id=authority.controller_id,
                                           command_sequence=sequence, nonce=nonce)
    pointer = run / 'distributed-checkpoints' / 'latest.json'
    pointer.write_text('{"previous": "durable"}')
    return authority, staging, receipt, pointer


def test_parent_commit_is_torch_free_and_consumes_sequence(tmp_path):
    authority, staging, receipt, pointer = _fixture(tmp_path)
    with run_lock(tmp_path):
        target = authority.commit(receipt, expected_command_sequence=1)
        assert not staging.exists()
        assert json.loads(pointer.read_text())['checkpoint'] == target.name
        assert (target / 'rank-00001.pt').read_bytes() == b'opaque rank 1'
        with pytest.raises(ValueError, match='consumed'):
            authority.commit(receipt, expected_command_sequence=1)
        authority.close()
    program = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'particlegan'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import hypergan.distributed_commit
'''
    result = subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', program], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('field,value', [('run_id', 'old-run'), ('attempt_id', 'old-attempt'),
                                        ('controller_id', 'old-controller'), ('command_sequence', 2),
                                        ('command_sequence', True), ('staging', '../escape'),
                                        ('checkpoint', '../escape'), ('identity_sha256', 'b' * 64)])
def test_stale_or_forged_receipt_never_changes_latest(tmp_path, field, value):
    authority, staging, receipt, pointer = _fixture(tmp_path)
    before = pointer.read_bytes()
    receipt[field] = value
    with pytest.raises(ValueError):
        authority.commit(receipt, expected_command_sequence=1)
    assert pointer.read_bytes() == before and staging.is_dir()


def test_takeover_and_process_fences(tmp_path, monkeypatch):
    authority, staging, receipt, pointer = _fixture(tmp_path)
    replacement = commit.CheckpointCommitAuthority(tmp_path, run_id='run-one', attempt_id='attempt-one',
                                                   identity={'config': {'training': {'steps': 3}}, 'topology': {'world_size': 2}})
    with pytest.raises(ValueError, match='stale'):
        replacement.commit(receipt, expected_command_sequence=1)
    pid = commit.os.getpid()
    with monkeypatch.context() as context:
        context.setattr(commit.os, 'getpid', lambda: pid + 1)
        with pytest.raises(RuntimeError, match='another process'):
            authority.commit(receipt, expected_command_sequence=1)
    authority.close()
    authority.close()
    with pytest.raises(RuntimeError, match='closed'):
        authority.commit(receipt, expected_command_sequence=1)


@pytest.mark.parametrize('mode', ['rank-corrupt', 'rank-missing', 'manifest-corrupt', 'extra-file', 'receipt-oversize', 'rank-oversize'])
def test_incomplete_or_corrupt_preparation_never_publishes(tmp_path, monkeypatch, mode):
    authority, staging, receipt, pointer = _fixture(tmp_path)
    before = pointer.read_bytes()
    if mode == 'rank-corrupt':
        path = staging / 'rank-00000.pt'
        path.write_bytes(b'!' * path.stat().st_size)
    elif mode == 'rank-missing':
        (staging / 'rank-00001.pt').unlink()
    elif mode == 'manifest-corrupt':
        path = staging / 'manifest.json'
        path.write_bytes(b'!' * path.stat().st_size)
    elif mode == 'extra-file':
        (staging / 'unexpected').write_text('extra')
    elif mode == 'receipt-oversize':
        monkeypatch.setattr(commit, 'MAX_RECEIPT_BYTES', 16)
    elif mode == 'rank-oversize':
        monkeypatch.setattr(commit, 'MAX_RANK_BYTES', 4)
    with pytest.raises(ValueError):
        authority.commit(receipt, expected_command_sequence=1)
    assert pointer.read_bytes() == before and staging.exists()
    assert not (pointer.parent / receipt['checkpoint']).exists()


@pytest.mark.parametrize('stage', ['rename', 'pointer', 'after-pointer'])
def test_publication_failure_consumes_receipt_and_preserves_honest_commit_state(tmp_path, monkeypatch, stage):
    authority, staging, receipt, pointer = _fixture(tmp_path)
    before = pointer.read_bytes()
    if stage == 'rename':
        original = Path.rename
        def broken(path, target):
            if path == staging:
                raise OSError('rename unavailable')
            return original(path, target)
        monkeypatch.setattr(Path, 'rename', broken)
    else:
        original = commit.atomic_json
        def broken(path, value):
            if stage == 'after-pointer':
                original(path, value)
            raise OSError('pointer durability unavailable')
        monkeypatch.setattr(commit, 'atomic_json', broken)
    with pytest.raises(OSError):
        authority.commit(receipt, expected_command_sequence=1)
    with pytest.raises(ValueError, match='consumed'):
        authority.commit(receipt, expected_command_sequence=1)
    if stage == 'after-pointer':
        assert json.loads(pointer.read_text())['checkpoint'] == receipt['checkpoint']
    else:
        assert pointer.read_bytes() == before
    assert staging.exists() == (stage == 'rename')
    assert (pointer.parent / receipt['checkpoint']).exists() == (stage != 'rename')


def test_both_rename_parent_directories_are_synced(tmp_path, monkeypatch):
    authority, staging, receipt, pointer = _fixture(tmp_path)
    paths = []
    original = commit.sync_directory
    def record(path):
        paths.append(Path(path))
        original(path)
    monkeypatch.setattr(commit, 'sync_directory', record)
    authority.commit(receipt, expected_command_sequence=1)
    assert paths[:2] == [staging.parent, pointer.parent]


@pytest.mark.parametrize('identity', [{}, {'config': {'training': {'steps': True}}, 'topology': {'world_size': 2}},
                                     {'config': {'training': {'steps': 3}}, 'topology': {'world_size': True}}])
def test_invalid_expected_identity_is_actionable(tmp_path, identity):
    with pytest.raises(ValueError, match='identity requires'):
        commit.CheckpointCommitAuthority(tmp_path, run_id='run-one', attempt_id='attempt-one', identity=identity)
