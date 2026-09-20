"""Torch-free parent authority for publishing prepared distributed checkpoints.

The caller retains the run lock throughout the authority lifetime. This object
does not acquire or verify that OS lock. A receipt is not proof that a supervised
command succeeded: check every worker's completion and current group health
before committing. Trusted workers in the new path only call prepare; this is
cooperative protocol fencing, not a sandbox against arbitrary Python file writes.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

from .run_state import atomic_json, sync_directory

SCHEMA = 1
KIND = 'hypergan-distributed-training-checkpoint'
PREPARED_KIND = 'hypergan-prepared-distributed-checkpoint'
MAX_RANK_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_BYTES = 65536
MAX_RANKS = 64
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
_HASH = re.compile(r'[a-f0-9]{64}\Z')
_NONCE = re.compile(r'[a-f0-9]{12}\Z')


def _json_bytes(value, limit=MAX_METADATA_BYTES):
    try:
        result = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('Checkpoint descriptor must contain finite JSON values') from exc
    if len(result) > limit:
        raise ValueError('Checkpoint descriptor exceeds its byte bound')
    return result


def identity_sha256(identity):
    return hashlib.sha256(_json_bytes(identity)).hexdigest()


def validate_fence(controller_id, command_sequence):
    if not isinstance(controller_id, str) or not _ID.fullmatch(controller_id):
        raise ValueError('Checkpoint controller_id must be a safe nonempty identifier')
    if type(command_sequence) is not int or not 1 <= command_sequence < 2 ** 63:
        raise ValueError('Checkpoint command_sequence must be a positive signed-64-bit integer')


def _directory(path):
    try:
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise ValueError(f'Checkpoint directory must be ordinary, not a symlink: {path}')
    except FileNotFoundError as exc:
        raise ValueError(f'Checkpoint directory is missing: {path}') from exc
    return path


def checkpoint_root(run_dir):
    run = Path(run_dir).resolve()
    _directory(run)
    root = run / 'distributed-checkpoints'
    if root.exists() or root.is_symlink():
        _directory(root)
    return root


def preparation_directory(run_dir, attempt_id, controller_id, command_sequence, nonce, *, create=False):
    """Resolve only a fixed managed path; never follow managed directory links."""
    validate_fence(controller_id, command_sequence)
    if not isinstance(attempt_id, str) or not _ID.fullmatch(attempt_id):
        raise ValueError('Checkpoint attempt_id must be a safe nonempty identifier')
    if not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
        raise ValueError('Invalid prepared checkpoint nonce')
    root = checkpoint_root(run_dir)
    current = root
    for component in (None, '.prepared', attempt_id, controller_id, f'command-{command_sequence:08d}-{nonce}'):
        if component is not None:
            current = current / component
        if create:
            try:
                current.mkdir()
                sync_directory(current.parent)
            except FileExistsError:
                if component and component.startswith('command-'):
                    raise ValueError('Prepared checkpoint directory already exists')
        _directory(current)
    return current


def _read_regular(path, *, expected_bytes=None, maximum=MAX_METADATA_BYTES, capture=False):
    """Read bounded ordinary files; hash payloads in chunks without loading torch."""
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Checkpoint files must be ordinary files, not links or special files')
    except FileNotFoundError as exc:
        raise ValueError(f'Checkpoint file is missing: {path.name}') from exc
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f'Checkpoint file cannot be opened: {path.name}') from exc
    with os.fdopen(descriptor, 'rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= maximum:
            raise ValueError('Checkpoint file has an invalid type or exceeds its byte bound')
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ValueError('Checkpoint file size differs from prepared descriptor')
        digest, blocks, count = hashlib.sha256(), [], 0
        while True:
            block = source.read(min(1048576, maximum - count + 1))
            if not block:
                break
            count += len(block)
            if count > maximum:
                raise ValueError('Checkpoint file grew beyond its byte bound')
            digest.update(block)
            if capture:
                blocks.append(block)
        after = os.fstat(source.fileno())
        if count != before.st_size or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            raise ValueError('Checkpoint file changed while being validated')
    return digest.hexdigest(), b''.join(blocks) if capture else None


def make_prepared_receipt(run_dir, staging, info, *, controller_id, command_sequence, nonce):
    """Build the transport descriptor after rank payload and manifest staging."""
    manifest = Path(staging) / 'manifest.json'
    digest, _ = _read_regular(manifest)
    receipt = {'schema_version': SCHEMA, 'kind': PREPARED_KIND,
               'run_id': info['run_id'], 'attempt_id': info['attempt_id'],
               'controller_id': controller_id, 'command_sequence': command_sequence,
               'step': info['step'], 'nonce': nonce,
               'staging': str(Path(staging).relative_to(Path(run_dir).resolve()).as_posix()),
               'checkpoint': f"{info['attempt_id']}-step-{info['step']:08d}-{nonce}",
               'manifest_bytes': manifest.stat().st_size, 'manifest_sha256': digest,
               'identity_sha256': identity_sha256(info['identity']), 'ranks': info['ranks']}
    _json_bytes(receipt, MAX_RECEIPT_BYTES)
    return receipt


def _validate_prepared(run_dir, receipt, *, run_id, attempt_id, controller_id, command_sequence, identity):
    _json_bytes(receipt, MAX_RECEIPT_BYTES)
    keys = {'schema_version', 'kind', 'run_id', 'attempt_id', 'controller_id', 'command_sequence', 'step',
            'nonce', 'staging', 'checkpoint', 'manifest_bytes', 'manifest_sha256', 'identity_sha256', 'ranks'}
    if not isinstance(receipt, dict) or set(receipt) != keys or type(receipt['schema_version']) is not int or receipt['schema_version'] != SCHEMA or receipt['kind'] != PREPARED_KIND:
        raise ValueError('Invalid prepared checkpoint receipt schema')
    validate_fence(receipt['controller_id'], receipt['command_sequence'])
    if (receipt['run_id'], receipt['attempt_id'], receipt['controller_id'], receipt['command_sequence']) != (run_id, attempt_id, controller_id, command_sequence):
        raise ValueError('Prepared checkpoint has a stale run/attempt/controller/command fence')
    if receipt['identity_sha256'] != identity_sha256(identity):
        raise ValueError('Prepared checkpoint expected identity differs')
    step = receipt['step']
    if type(step) is not int or not 0 <= step <= identity['config']['training']['steps']:
        raise ValueError('Prepared checkpoint step is outside the original schedule')
    staging = preparation_directory(run_dir, attempt_id, controller_id, command_sequence, receipt['nonce'])
    if receipt['staging'] != staging.relative_to(Path(run_dir).resolve()).as_posix():
        raise ValueError('Prepared checkpoint staging path is outside its managed fence')
    root = checkpoint_root(run_dir)
    name = f"{attempt_id}-step-{step:08d}-{receipt['nonce']}"
    if receipt['checkpoint'] != name:
        raise ValueError('Prepared checkpoint destination path differs from its identity')
    target = root / name
    if target.exists() or target.is_symlink():
        raise ValueError('Immutable checkpoint destination already exists')
    if (root / 'latest.json').is_symlink():
        raise ValueError('Distributed checkpoint latest pointer must not be a symlink')
    if type(receipt['manifest_bytes']) is not int or not 1 <= receipt['manifest_bytes'] <= MAX_METADATA_BYTES:
        raise ValueError('Invalid prepared manifest byte bound')
    digest, content = _read_regular(staging / 'manifest.json', expected_bytes=receipt['manifest_bytes'], capture=True)
    if digest != receipt['manifest_sha256']:
        raise ValueError('Prepared checkpoint manifest digest mismatch')
    try:
        info = json.loads(content)
    except (ValueError, RecursionError) as exc:
        raise ValueError('Invalid prepared checkpoint manifest JSON') from exc
    required = {'schema_version', 'kind', 'run_id', 'attempt_id', 'step', 'identity', 'replicated_sha256', 'ranks', 'next_sample_sequence'}
    if (not isinstance(info, dict) or not required <= set(info) <= required | {'request_ids', 'event_boundary', 'source', 'initial_source'}
            or type(info['schema_version']) is not int or info['schema_version'] != SCHEMA or info['kind'] != KIND):
        raise ValueError('Invalid prepared checkpoint manifest schema')
    if any(key in info and not isinstance(info[key], dict) for key in ('source', 'initial_source')):
        raise ValueError('Prepared checkpoint source provenance must be a dictionary')
    if (info['run_id'], info['attempt_id']) != (run_id, attempt_id) or type(info['step']) is not int or info['step'] != step:
        raise ValueError('Prepared checkpoint manifest lineage/step differs')
    if _json_bytes(info['identity']) != _json_bytes(identity):
        raise ValueError('Prepared checkpoint manifest identity differs')
    if type(info['next_sample_sequence']) is not int or info['next_sample_sequence'] < 1:
        raise ValueError('Invalid prepared sample sequence')
    ids = info.get('request_ids', [])
    if not isinstance(ids, list) or len(ids) > 256 or any(not isinstance(value, str) or not _ID.fullmatch(value) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Invalid prepared checkpoint request IDs')
    if not isinstance(info['replicated_sha256'], str) or not _HASH.fullmatch(info['replicated_sha256']):
        raise ValueError('Invalid prepared replicated-state digest')
    world = identity['topology']['world_size']
    if type(world) is not int or not 2 <= world <= MAX_RANKS:
        raise ValueError('Prepared checkpoint requires between 2 and 64 fixed ranks')
    if not isinstance(info['ranks'], list) or len(info['ranks']) != world or _json_bytes(info['ranks']) != _json_bytes(receipt['ranks']):
        raise ValueError('Prepared checkpoint must contain exactly the agreed rank inventory')
    expected_files = {'manifest.json'}
    for rank, record in enumerate(info['ranks']):
        if (not isinstance(record, dict) or set(record) != {'rank', 'file', 'bytes', 'sha256'}
                or type(record['rank']) is not int or record['rank'] != rank or record['file'] != f'rank-{rank:05d}.pt'):
            raise ValueError('Invalid prepared rank inventory')
        if type(record['bytes']) is not int or not 1 <= record['bytes'] <= MAX_RANK_BYTES:
            raise ValueError('Prepared rank payload exceeds its byte bound')
        if not isinstance(record['sha256'], str) or not _HASH.fullmatch(record['sha256']):
            raise ValueError('Invalid prepared rank digest')
        actual, _ = _read_regular(staging / record['file'], expected_bytes=record['bytes'], maximum=MAX_RANK_BYTES)
        if actual != record['sha256']:
            raise ValueError(f'Prepared rank {rank} payload digest mismatch')
        expected_files.add(record['file'])
    names = set()
    with os.scandir(staging) as entries:
        for entry in entries:
            names.add(entry.name)
            if len(names) > len(expected_files):
                break
    if names != expected_files:
        raise ValueError('Prepared generation contains unexpected or incomplete files')
    return staging, target, info


def _publish(staging, target, info):
    # The caller has validated every byte and owns the sole canonical writer.
    # Rename success followed by a pointer/fsync failure may leave a complete,
    # unselected generation. A post-replace fsync failure may leave latest changed.
    source_parent = staging.parent
    staging.rename(target)
    sync_directory(source_parent)
    sync_directory(target.parent)
    atomic_json(target.parent / 'latest.json', {'schema_version': SCHEMA, 'kind': KIND,
                                              'checkpoint': target.name, 'step': info['step']})
    return target


class CheckpointCommitAuthority:
    """Parent/PID-scoped publication fence, used while the caller holds run_lock.

    Only pass ``controller_id`` to workers. Never send this authority to them.
    Call ``commit`` only after successful all-rank command completion and a fresh
    supervisor health check. A sequence is consumed after validation and before
    any rename: uncertain write failures cannot retry that receipt. Start a new
    attempt after takeover; closing the authority invalidates pending receipts.
    """
    def __init__(self, run_dir, *, run_id, attempt_id, identity):
        for name, value in (('run_id', run_id), ('attempt_id', attempt_id)):
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise ValueError(f'Invalid checkpoint authority {name}')
        self._root = checkpoint_root(run_dir)
        self._run_id, self._attempt_id = run_id, attempt_id
        self._identity = json.loads(_json_bytes(identity))
        config = self._identity.get('config') if isinstance(self._identity, dict) else None
        training = config.get('training') if isinstance(config, dict) else None
        topology = self._identity.get('topology') if isinstance(self._identity, dict) else None
        if not isinstance(training, dict) or type(training.get('steps')) is not int or training['steps'] < 0:
            raise ValueError('Checkpoint authority identity requires config.training.steps as a nonnegative integer')
        if not isinstance(topology, dict) or type(topology.get('world_size')) is not int or not 2 <= topology['world_size'] <= MAX_RANKS:
            raise ValueError('Checkpoint authority identity requires topology.world_size between 2 and 64')
        self._pid = os.getpid()
        self._controller_id = uuid.uuid4().hex
        self._consumed_sequence = 0
        self._closed = False

    @property
    def controller_id(self):
        return self._controller_id

    def _active(self):
        if os.getpid() != self._pid or self._closed:
            raise RuntimeError('Checkpoint commit authority belongs to another process or is closed')

    def commit(self, receipt, *, expected_command_sequence, expected_event_boundary=None):
        self._active()
        validate_fence(self._controller_id, expected_command_sequence)
        if expected_command_sequence <= self._consumed_sequence:
            raise ValueError('Checkpoint command sequence is stale or already consumed')
        staging, target, info = _validate_prepared(self._root.parent, receipt,
            run_id=self._run_id, attempt_id=self._attempt_id, controller_id=self._controller_id,
            command_sequence=expected_command_sequence, identity=self._identity)
        if expected_event_boundary is not None and _json_bytes(info.get('event_boundary')) != _json_bytes(expected_event_boundary):
            raise ValueError('Prepared checkpoint event boundary differs from the controller durable prefix')
        self._consumed_sequence = expected_command_sequence
        return _publish(staging, target, info)

    def close(self):
        if os.getpid() != self._pid:
            raise RuntimeError('Checkpoint commit authority belongs to another process')
        self._closed = True

    def __enter__(self):
        self._active()
        return self

    def __exit__(self, *unused):
        self.close()
