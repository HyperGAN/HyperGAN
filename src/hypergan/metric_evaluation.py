"""Manual and interval snapshot evaluation against immutable EMA bundles.

The public host is torch-free. The standalone command does not take the run
lock, so it can run beside training; it holds an owner lock on its own
evaluation directory instead, which keeps the training process from recovering
it as abandoned. Interval evaluation runs under its training coordinator's
lock. Both paths use a fresh supervised worker, then atomically register an
independent result stream. No trainer state, training sampler, or training
event writer is reused.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import time
import uuid

from foldlitedb.fileio import file_lock

from .config import fingerprint, load_config
from .metric_plugins import enabled_custom, finite_json, prepare_custom
from .metrics import digest, metric_catalog
from .run_state import atomic_json

# Large transformer generators include position buffers as well as parameters.
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
_EVAL_DIR_RE = re.compile(r'^(?:step-\d+-[A-Za-z0-9_.-]+|[0-9a-f]{32})$')
# Held by a manual evaluation for its whole lifetime; the OS releases it on death.
OWNER_LOCK = '.owner.lock'


def _normalize_eval_device(dev):
    if isinstance(dev, int):
        return f"cuda:{dev}"
    dev_str = str(dev).strip().lower()
    if dev_str.isdigit():
        return f"cuda:{dev_str}"
    if dev_str.startswith("gpu") and dev_str[3:].isdigit():
        return f"cuda:{dev_str[3:]}"
    if dev_str == "cpu":
        return "cpu"
    if dev_str == "cuda":
        return "cuda:0"
    return dev_str


def _parse_device_set(dev):
    if dev is None:
        return {"cuda:0"}
    if isinstance(dev, (list, tuple, set)):
        result = set()
        for item in dev:
            result.update(_parse_device_set(item))
        return result
    if isinstance(dev, int):
        return {f"cuda:{dev}"}
    dev_str = str(dev).strip()
    if not dev_str:
        return {"cuda:0"}
    if "," in dev_str:
        result = set()
        for part in dev_str.split(","):
            part = part.strip()
            if part:
                result.update(_parse_device_set(part))
        return result
    return {_normalize_eval_device(dev_str)}


def _devices_conflict(d1, d2):
    s1 = _parse_device_set(d1)
    s2 = _parse_device_set(d2)
    s1.discard("cpu")
    s2.discard("cpu")
    return bool(s1 & s2)


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


def evaluate(run_dir, metric_id=None, *, checkpoint=None, devices=None, config_path=None, bundle=None):
    """Manually evaluate one configured snapshot metric of a run.

    It never changes training state, but writes its own evaluation directory,
    the metric catalog and summary.json. It runs beside training without the
    run lock, holding only its evaluation directory's owner lock.

    Each invocation starts a new evaluation ID. A failure publishes a failed
    receipt and source event; on_error=fail then raises with that receipt path.
    """
    import torch
    root = Path(run_dir).resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f"Run directory missing manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    allowed_statuses = ('running', 'complete', 'stopped', 'failed', 'interrupted', 'initializing')
    if manifest.get('status') not in allowed_statuses:
        raise ValueError(f"Cannot evaluate run with status: {manifest.get('status')}")

    config = load_config(config_path) if config_path is not None else load_config(manifest['config'])
    if fingerprint(config) != manifest['config_sha256']:
        raise ValueError('Evaluation configuration changes the numerical recipe')

    specs = enabled_custom(config)
    snapshot_specs = {k: v for k, v in specs.items() if v.get('mode') == 'snapshot'}
    if not snapshot_specs:
        raise ValueError('No custom snapshot metrics enabled in configuration')

    if metric_id is None:
        if len(snapshot_specs) == 1:
            metric_id = next(iter(snapshot_specs))
        else:
            raise ValueError(f'--metric required; available: {sorted(list(snapshot_specs.keys()))}')
    elif metric_id not in snapshot_specs:
        raise ValueError(f'Metric {metric_id!r} not enabled with mode="snapshot" (available: {sorted(list(snapshot_specs.keys()))})')

    selected = dict(config)
    selected['metrics'] = dict(config['metrics'], preset='none', custom={metric_id: snapshot_specs[metric_id]},
                               disable=[], overrides={})
    prepare_custom(selected)
    spec = snapshot_specs[metric_id]

    warnings_list = []
    if devices is not None:
        eval_device = _normalize_eval_device(devices)
        spec.setdefault('evaluation', {})['device'] = eval_device
    else:
        eval_device = spec.get('evaluation', {}).get('device', 'cuda')

    train_device = manifest.get('runtime', {}).get('device') or config.get('training', {}).get('device', 'cuda')
    if manifest.get('status') == 'running' and _devices_conflict(eval_device, train_device):
        msg = f"warning: evaluation shares device '{eval_device}' with active training run; may cause GPU contention"
        sys.stderr.write(msg + '\n')
        warnings_list.append({
            'code': 'DEVICE_COLLISION',
            'message': msg,
            'evaluation_device': eval_device,
            'training_device': train_device,
        })

    # Resolve bundle or checkpoint
    from .checkpoints import resolve_checkpoint, extract_ema_bundle
    source_file = None
    bundle_dict = None
    step = 0
    attempt_id = manifest.get('attempt_id', 'eval')

    if bundle is not None:
        candidate = Path(bundle)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_dir():
            candidate = candidate / 'model.pt'
        if not candidate.is_file():
            raise ValueError(f'Bundle not found: {candidate}')
        source_file = candidate.resolve()
        bundle_dict = torch.load(source_file, map_location='cpu', weights_only=True)
        step = bundle_dict.get('step', 0)
        attempt_id = bundle_dict.get('identity', {}).get('attempt_id', attempt_id)

    if bundle_dict is None:
        target = checkpoint if checkpoint is not None else 'latest'
        checkpoint_dir = resolve_checkpoint(root, target)
        state_file = checkpoint_dir / 'state.pt'
        model_file = checkpoint_dir / 'model.pt'
        if state_file.is_file():
            source_file = state_file
            ckpt_manifest_path = checkpoint_dir / 'manifest.json'
            ckpt_manifest = json.loads(ckpt_manifest_path.read_text(encoding='utf-8')) if ckpt_manifest_path.is_file() else manifest
            step = ckpt_manifest.get('step', 0)
            attempt_id = ckpt_manifest.get('attempt_id', attempt_id)
            bundle_dict = extract_ema_bundle(checkpoint_dir, manifest=ckpt_manifest)
        elif model_file.is_file():
            source_file = model_file
            bundle_dict = torch.load(model_file, map_location='cpu', weights_only=True)
            step = bundle_dict.get('step', 0)
            attempt_id = bundle_dict.get('identity', {}).get('attempt_id', attempt_id)
        else:
            raise ValueError(f"No state.pt or model.pt in checkpoint {checkpoint_dir}")

    sanitized_metric = re.sub(r'[^A-Za-z0-9_-]', '_', metric_id)
    short_uuid = uuid.uuid4().hex[:8]
    dir_name = f"step-{step:08d}-{sanitized_metric}-{short_uuid}"
    bundle_dict.setdefault('identity', {})
    bundle_dict['identity']['evaluation_id'] = dir_name
    if not bundle_dict['identity'].get('attempt_id'):
        bundle_dict['identity']['attempt_id'] = attempt_id
    if not bundle_dict['identity'].get('run_id'):
        bundle_dict['identity']['run_id'] = manifest['run_id']
    directory = root / 'metrics' / 'evaluations' / dir_name
    directory.mkdir(parents=True, exist_ok=False)
    with file_lock(directory, name=OWNER_LOCK):
        return _evaluate_owned(root, manifest, selected, metric_id, directory, dir_name, step,
                               attempt_id, source_file, bundle_dict, warnings_list)


def _evaluate_owned(root, manifest, selected, metric_id, directory, evaluation_id, step,
                    attempt_id, source_file, bundle_dict, warnings_list):
    import torch
    snapshot = directory / 'snapshot.pt'
    torch.save(bundle_dict, snapshot)
    expected_hash = _sha256(snapshot)

    source_bundle_str = source_file.relative_to(root).as_posix() if source_file and source_file.is_relative_to(root) else str(source_file or "")
    identity = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                'evaluation_id': evaluation_id, 'source_step': step,
                'source_bundle': source_bundle_str}
    if 'attempt_index' in bundle_dict['identity']:
        identity['attempt_index'] = bundle_dict['identity']['attempt_index']
    initial_catalog = _write_catalog(root, metric_catalog(selected))
    receipt_running = {
        'schema_version': 1, 'status': 'running', **identity,
        'snapshot_sha256': expected_hash, 'metric_id': metric_id,
        'catalog': initial_catalog,
    }
    if warnings_list:
        receipt_running['warnings'] = warnings_list
    atomic_json(directory / 'receipt.json', receipt_running)

    try:
        return _evaluate_pinned(root, selected, metric_id, directory, snapshot, expected_hash, identity, warnings=warnings_list)
    finally:
        snapshot.unlink(missing_ok=True)


def _evaluate_pinned(root, selected, metric_id, directory, snapshot, expected_hash, identity, *, cancellation_event=None, timeout=None, warnings=None):
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
    if warnings:
        receipt['warnings'] = warnings
    atomic_json(directory / 'receipt.json', receipt)
    atomic_json(directory / 'stream.json', {'schema_version': 1,
        'stream_id': event['stream_id'], 'stream_generation': evaluation_id,
        'run_id': manifest['run_id'], 'path': (directory / 'events.jsonl').relative_to(root).as_posix(),
        'role': 'measurement', 'modality': descriptor['kind']})
    if status == 'complete':
        try:
            from .summary import write_summary_json
            write_summary_json(root)
        except Exception:
            pass
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        raise failure
    if status == 'failed' and spec['on_error'] == 'fail':
        raise RuntimeError(f'Metric evaluation failed; receipt: {directory / "receipt.json"}: {result["error"]}') from failure
    return receipt


def _owner_alive(path):
    """Whether a manual evaluation still holds this directory's owner lock."""
    if not (path / OWNER_LOCK).is_file():
        return False
    try:
        with file_lock(path, name=OWNER_LOCK):
            return False
    except RuntimeError:
        return True


def _recover_abandoned(root, run_id=None):
    """Run-lock ownership proves no previous interval evaluator for this run is active.

    A manual evaluation runs without the run lock, so its directory is skipped
    while its owner lock is held. Completed registered results are immutable. A
    stopped coordinator's pending evaluation is a failed observation, never a
    resumable partial accumulator.
    """
    root = Path(root)
    if (root / 'metrics').is_symlink():
        raise ValueError('Metrics directory must not be a symlink')
    directory = root / 'metrics' / 'evaluations'
    if not directory.exists():
        return
    if directory.is_symlink():
        raise ValueError('Evaluation directory must not be a symlink')
    if run_id is None:
        manifest_path = root / 'manifest.json'
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding='utf-8'))
                run_id = manifest_data.get('run_id')
            except Exception:
                pass
    for path in directory.iterdir():
        if not _EVAL_DIR_RE.match(path.name):
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Evaluation entries must be ordinary directories')
        receipt_path = path / 'receipt.json'
        if (path / 'stream.json').exists() or _owner_alive(path):
            continue
        if not receipt_path.exists():
            (path / 'snapshot.pt').unlink(missing_ok=True)
            continue
        if receipt_path.is_symlink() or receipt_path.stat().st_size > 65536:
            raise ValueError('Evaluation receipt must be a bounded ordinary file')
        receipt = json.loads(receipt_path.read_text())
        effective_run_id = run_id if run_id is not None else receipt.get('run_id')
        if run_id is not None and receipt.get('run_id') != run_id:
            raise ValueError('Abandoned evaluation identity differs from its run')

        eval_id = str(receipt.get('evaluation_id', ''))
        step = receipt.get('source_step')
        expected_names = {eval_id}
        if eval_id:
            if isinstance(step, int):
                expected_names.add(f"step-{step:08d}-{eval_id}")
            if path.name.endswith(f"-{eval_id}"):
                expected_names.add(path.name)
        if path.name not in expected_names:
            raise ValueError('Abandoned evaluation identity differs from its run')

        if receipt.get('status') in ('complete', 'failed', 'cancelled'):
            event_path = path / 'events.jsonl'
            if event_path.is_symlink() or not event_path.is_file() or event_path.stat().st_size > 65536:
                raise ValueError('Unregistered completed evaluation requires a bounded event file')
            encoded = event_path.read_bytes()
            event = json.loads(encoded)
            if (hashlib.sha256(encoded).hexdigest() != receipt.get('event_sha256')
                    or (run_id is not None and event.get('run_id') != run_id)
                    or event.get('evaluation_id') not in expected_names):
                raise ValueError('Unregistered evaluation event integrity mismatch')
            atomic_json(path / 'stream.json', {'schema_version': 1, 'stream_id': 'evaluation:' + path.name,
                'stream_generation': path.name, 'run_id': effective_run_id,
                'path': event_path.relative_to(root).as_posix(), 'role': 'measurement',
                'modality': 'histogram' if event.get('distributions') else 'scalar'})
            (path / 'snapshot.pt').unlink(missing_ok=True)
            continue
        if receipt.get('status') != 'running':
            raise ValueError('Unsupported evaluation receipt status')
        reason = 'Evaluation ended before result publication; restart explicitly with a new evaluation ID'
        event = {'schema_version': 2, 'event': 'evaluation', 'run_id': effective_run_id,
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
            'stream_generation': path.name, 'run_id': effective_run_id,
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
            raise ValueError('Evaluation snapshot grew beyond the 512 MiB copy budget')
        destination.write(block)
