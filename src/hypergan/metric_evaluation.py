"""Manual and interval snapshot evaluation against immutable EMA bundles.

The public host is torch-free. The standalone command holds the run lock and pins bytes;
interval evaluation runs under its training coordinator's lock. Both paths
uses a fresh supervised worker, then atomically registers its independent result
stream. No trainer state, training sampler, or training event writer is reused.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import time
import uuid

from .config import fingerprint, load_config, resolve_config
from .metric_plugins import enabled_custom, finite_json, prepare_custom
from .metrics import digest, metric_catalog
from .run_state import atomic_json, run_lock, sync_directory

MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def _worker_factory(rank, world_size, spec, expected, snapshot, snapshot_sha256, identity):
    return spec, expected, snapshot, snapshot_sha256, identity


def _worker_command(state, operation, payload):
    if operation != 'evaluate':
        raise ValueError('Unknown snapshot metric operation')
    from .metric_evaluation_worker import evaluate_snapshot
    return evaluate_snapshot(*state)


def _write_catalog(root, catalog):
    revision = digest(catalog)
    path = root / 'metrics' / f'catalog-{revision}.json'
    if path.exists():
        from .metrics import read_catalog
        if read_catalog(root, revision) != catalog:
            raise ValueError('Existing catalog differs from evaluation definition')
    else:
        atomic_json(path, catalog)
    return revision


def evaluate(run_dir, metric_id, *, config_path=None, bundle=None):
    """Manually evaluate one configured snapshot metric; no retries or downloads.

    Each invocation starts a new evaluation ID. A failure publishes a failed
    receipt and source event; on_error=fail then raises with that receipt path.
    Training and evaluation cannot own this run simultaneously. The default
    evaluator device is CUDA; CPU correctness fixtures opt in in configuration.
    """
    root = Path(run_dir).resolve()
    with run_lock(root):
        manifest = json.loads((root / 'manifest.json').read_text())
        _recover_abandoned(root, manifest['run_id'])
        if manifest.get('status') not in ('complete', 'stopped', 'failed', 'interrupted'):
            raise ValueError('Standalone evaluation requires a terminal run; training and evaluation are serialized')
        config = load_config(config_path) if config_path is not None else resolve_config(manifest['config'])
        if fingerprint(config) != manifest['config_sha256']:
            raise ValueError('Evaluation configuration changes the numerical recipe')
        specs = enabled_custom(config)
        if metric_id not in specs or specs[metric_id]['mode'] != 'snapshot':
            raise ValueError('Select an enabled custom metric with mode="snapshot"')
        # Resolve only this selected factory; unrelated disabled/failed factories
        # cannot be executed as a side effect of a manual evaluation.
        selected = dict(config)
        selected['metrics'] = dict(config['metrics'], preset='none', custom={metric_id: specs[metric_id]},
                                   disable=[], overrides={})
        prepare_custom(selected)
        spec = specs[metric_id]
        candidate = Path(bundle or manifest.get('bundle_path', ''))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_dir():
            candidate = candidate / 'model.pt'
        path = candidate.resolve()
        if candidate != path or not path.is_relative_to(root / 'attempts') or not path.is_file():
            raise ValueError('Select an immutable EMA model.pt bundle inside this run attempts directory')
        if not 0 < path.stat().st_size <= MAX_SNAPSHOT_BYTES:
            raise ValueError('Evaluation snapshot exceeds the 256 MiB limit')
        checksum = path.parent / 'model.sha256'
        if checksum.is_symlink() or not checksum.is_file() or checksum.stat().st_size > 128:
            raise ValueError('EMA bundle checksum must be a bounded ordinary file')
        expected_hash = checksum.read_text().strip()
        if re.fullmatch(r'[0-9a-f]{64}', expected_hash) is None:
            raise ValueError('Invalid EMA bundle checksum')
        evaluation_id = uuid.uuid4().hex
        directory = root / 'metrics' / 'evaluations' / evaluation_id
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = directory / 'snapshot.pt'
        identity = {'run_id': manifest['run_id'], 'evaluation_id': evaluation_id,
                    'source_bundle': path.relative_to(root).as_posix()}
        initial_catalog = _write_catalog(root, metric_catalog(selected))
        atomic_json(directory / 'receipt.json', {'schema_version': 1, 'status': 'running', **identity,
                                               'snapshot_sha256': expected_hash, 'metric_id': metric_id,
                                               'catalog': initial_catalog})
        try:
            with path.open('rb') as source, snapshot.open('xb') as destination:
                _copy_snapshot(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            if snapshot.stat().st_size > MAX_SNAPSHOT_BYTES or _sha256(snapshot) != expected_hash:
                raise ValueError('EMA bundle bytes do not match their immutable checksum')
            return _evaluate_pinned(root, selected, metric_id, directory, snapshot, expected_hash, identity)
        finally:
            # Complete/failed results keep the immutable bundle digest/provenance,
            # not an extra copy of large model weights. Original bundle stays put.
            snapshot.unlink(missing_ok=True)
            _recover_abandoned(root, manifest['run_id'])


def _evaluate_pinned(root, selected, metric_id, directory, snapshot, expected_hash, identity, *, cancellation_event=None, timeout=None):
    """Evaluate and register owned immutable bytes; caller owns run coordination."""
    from .cpu_worker_service import CPUServiceCancelled, CPUWorkerService
    spec = selected['metrics']['custom'][metric_id]
    evaluation_id = identity['evaluation_id']
    timeout = spec['timeout'] if timeout is None else timeout
    manifest = {'run_id': identity['run_id']}
    started = time.monotonic()
    service = CPUWorkerService(_worker_factory, _worker_command,
        args=(spec, selected['_metric_runtime'][metric_id], str(snapshot), expected_hash, identity),
        run_id=manifest['run_id'], attempt_id=evaluation_id, world_size=1, initialize_process_group=False,
        cancellation_event=cancellation_event,
        startup_timeout=timeout, command_timeout=timeout,
        collective_timeout=timeout, total_timeout=timeout)
    failure = None
    try:
        with service:
            result = finite_json(service.command('evaluate')['results'][0])
        status = 'complete'
    except BaseException as error:
        failure = error
        status = 'cancelled' if isinstance(error, CPUServiceCancelled) else 'failed'
        result = {'error': f'{type(error).__name__}: {error}'[:1000]}
    elapsed = time.monotonic() - started
    catalog = metric_catalog(selected)
    descriptor = catalog['metrics'][metric_id]
    if status == 'complete':
        descriptor['evaluation_protocol'] = result['protocol']
        descriptor['definition_hash'] = digest({k: v for k, v in descriptor.items() if k != 'definition_hash'})
    revision = _write_catalog(root, catalog)
    # The pinned bundle's owning attempt and update are read by the
    # numerical worker. Failures before load have explicitly unknown
    # source position and cannot be mistaken for current-run measures.
    known_source = type(identity.get('source_step')) is int and identity.get('attempt_id') is not None
    owner = result.get('snapshot_identity', identity if known_source else {})
    event = {'schema_version': 2, 'event': 'evaluation', 'run_id': manifest['run_id'],
             'attempt_id': owner.get('attempt_id', evaluation_id), 'sequence': 1,
             'stream_id': 'evaluation:' + evaluation_id, 'stream_generation': evaluation_id,
             'step': result.get('step', identity.get('source_step', 0)), 'seconds': elapsed, 'catalog': revision,
             'metrics': {}, 'measurement_status': {}, 'evaluation_id': evaluation_id,
             'snapshot_sha256': expected_hash, 'snapshot_identity': owner, 'status': status,
             'source_position_known': status == 'complete' or known_source}
    if status == 'complete':
        target = 'metrics' if descriptor['kind'] == 'scalar' else 'distributions'
        event.setdefault(target, {})[metric_id] = result['value']
        event['evaluation_protocol'] = result['protocol']
        event['protocol_sha256'] = digest(result['protocol'])
    else:
        event['measurement_status'][metric_id] = {'status': status, 'reason': result['error']}
    encoded = json.dumps(event, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(encoded) > 65536:
        raise ValueError('Evaluation event exceeds 64 KiB')
    with (directory / 'events.jsonl').open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {'schema_version': 1, 'status': status, **identity, 'metric_id': metric_id,
               'catalog': revision, 'snapshot_sha256': expected_hash, 'seconds': elapsed,
               'result': result, 'event_sha256': hashlib.sha256(encoded).hexdigest()}
    atomic_json(directory / 'receipt.json', receipt)
    atomic_json(directory / 'stream.json', {'schema_version': 1,
        'stream_id': event['stream_id'], 'stream_generation': evaluation_id,
        'run_id': manifest['run_id'], 'path': (directory / 'events.jsonl').relative_to(root).as_posix(),
        'role': 'measurement', 'modality': descriptor['kind']})
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        raise failure
    if status == 'failed' and spec['on_error'] == 'fail':
        raise RuntimeError(f'Metric evaluation failed; receipt: {directory / "receipt.json"}: {result["error"]}') from failure
    return receipt


def _recover_abandoned(root, run_id):
    """Run-lock ownership proves no previous evaluator for this run is active.

    Completed registered results are immutable. A stopped coordinator's pending
    evaluation is a failed observation, never a resumable partial accumulator.
    """
    if (root / 'metrics').is_symlink():
        raise ValueError('Metrics directory must not be a symlink')
    directory = root / 'metrics' / 'evaluations'
    if not directory.exists():
        return
    if directory.is_symlink():
        raise ValueError('Evaluation directory must not be a symlink')
    for path in directory.iterdir():
        if re.fullmatch(r'[0-9a-f]{32}', path.name) is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Evaluation entries must be ordinary directories')
        receipt_path = path / 'receipt.json'
        if (path / 'stream.json').exists():
            continue
        if not receipt_path.exists():
            (path / 'snapshot.pt').unlink(missing_ok=True)
            continue
        if receipt_path.is_symlink() or receipt_path.stat().st_size > 65536:
            raise ValueError('Evaluation receipt must be a bounded ordinary file')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('run_id') != run_id or receipt.get('evaluation_id') != path.name:
            raise ValueError('Abandoned evaluation identity differs from its run')
        if receipt.get('status') in ('complete', 'failed', 'cancelled'):
            event_path = path / 'events.jsonl'
            if event_path.is_symlink() or not event_path.is_file() or event_path.stat().st_size > 65536:
                raise ValueError('Unregistered completed evaluation requires a bounded event file')
            encoded = event_path.read_bytes()
            event = json.loads(encoded)
            if (hashlib.sha256(encoded).hexdigest() != receipt.get('event_sha256')
                    or event.get('run_id') != run_id or event.get('evaluation_id') != path.name):
                raise ValueError('Unregistered evaluation event integrity mismatch')
            atomic_json(path / 'stream.json', {'schema_version': 1, 'stream_id': event['stream_id'],
                'stream_generation': path.name, 'run_id': run_id,
                'path': event_path.relative_to(root).as_posix(), 'role': 'measurement',
                'modality': 'histogram' if event.get('distributions') else 'scalar'})
            (path / 'snapshot.pt').unlink(missing_ok=True)
            continue
        if receipt.get('status') != 'running':
            raise ValueError('Unsupported evaluation receipt status')
        reason = 'Evaluation ended before result publication; restart explicitly with a new evaluation ID'
        event = {'schema_version': 2, 'event': 'evaluation', 'run_id': run_id,
                 'attempt_id': receipt.get('attempt_id', path.name), 'sequence': 1, 'stream_id': 'evaluation:' + path.name,
                 'stream_generation': path.name, 'catalog': receipt['catalog'],
                 'step': receipt.get('source_step', 0), 'seconds': 0.0, 'metrics': {}, 'status': 'failed',
                 'source_position_known': type(receipt.get('source_step')) is int, 'evaluation_id': path.name,
                 'snapshot_sha256': receipt['snapshot_sha256'], 'snapshot_identity': {},
                 'measurement_status': {receipt['metric_id']: {'status': 'failed', 'reason': reason}}}
        # No registry exists: any interrupted pre-registration event is not an
        # externally committed result. Publish one explicit failed replacement.
        event_path = path / 'events.jsonl'
        if event_path.is_symlink():
            raise ValueError('Evaluation event file must not be a symlink')
        encoded = json.dumps(event, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        temporary = path / ('.recovered-' + uuid.uuid4().hex + '.jsonl')
        with temporary.open('xb') as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, event_path)
        receipt.update(status='failed', result={'error': reason}, event_sha256=hashlib.sha256(encoded).hexdigest())
        atomic_json(receipt_path, receipt)
        atomic_json(path / 'stream.json', {'schema_version': 1, 'stream_id': event['stream_id'],
            'stream_generation': path.name, 'run_id': run_id,
            'path': event_path.relative_to(root).as_posix(), 'role': 'measurement', 'modality': 'scalar'})
        (path / 'snapshot.pt').unlink(missing_ok=True)


def _copy_snapshot(source, destination):
    copied = 0
    while True:
        block = source.read(min(1024 * 1024, MAX_SNAPSHOT_BYTES - copied + 1))
        if not block:
            return copied
        copied += len(block)
        if copied > MAX_SNAPSHOT_BYTES:
            raise ValueError('Evaluation snapshot grew beyond the 256 MiB copy budget')
        destination.write(block)
