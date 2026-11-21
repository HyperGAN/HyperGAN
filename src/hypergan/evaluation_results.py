"""Atomic publication and import of one set of snapshot scores."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .metric_plugins import METRIC_ID, finite_json, scalar_outputs, scalar_values
from .metrics import digest, read_catalog
from .run_state import atomic_json, sync_directory


def write_catalog(root, catalog):
    revision = digest(catalog)
    path = root / 'metrics' / f'catalog-{revision}.json'
    if path.exists():
        if read_catalog(root, revision) != catalog:
            raise ValueError('Existing catalog differs from evaluation definition')
    else:
        atomic_json(path, catalog)
    return revision


def write_result(directory, event, receipt, modality):
    """Register the stream only after both the event and its receipt are durable."""
    encoded = json.dumps(event, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(encoded) > 65536:
        raise ValueError('Evaluation event exceeds 64 KiB')
    with (directory / 'events.jsonl').open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    receipt['event_sha256'] = hashlib.sha256(encoded).hexdigest()
    atomic_json(directory / 'receipt.json', receipt)
    evaluation_id = event['evaluation_id']
    atomic_json(directory / 'stream.json', {'schema_version': 1, 'stream_id': event['stream_id'],
        'stream_generation': evaluation_id, 'run_id': event['run_id'],
        'path': f'metrics/evaluations/{evaluation_id}/events.jsonl', 'role': 'measurement', 'modality': modality})


def publish_scores(root, evaluator, event, receipt, *, running=None):
    """Commit one immutable stream per source position, evaluator and numerical protocol."""
    position = [event['run_id'], event['attempt_id'], event['step'], evaluator, event['protocol_sha256']]
    evaluation_id = f"step-{event['step']:08d}-{digest(position)[:32]}"
    event.update(evaluation_id=evaluation_id, stream_generation=evaluation_id, stream_id='evaluation:' + evaluation_id)
    receipt['evaluation_id'] = evaluation_id
    parent = root / 'metrics' / 'evaluations'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / evaluation_id
    if running is not None:
        # Recovery can distinguish a committed evaluation from an interrupted worker
        # if the process dies between publication and retiring its running receipt.
        path = running / 'receipt.json'
        atomic_json(path, dict(json.loads(path.read_text()), published_as=evaluation_id))
    with tempfile.TemporaryDirectory(prefix='.scores-', dir=parent.parent) as temporary:
        stage = Path(temporary) / 'result'
        stage.mkdir()
        write_result(stage, event, receipt, 'scalar')
        try:
            stage.rename(target)
        except OSError:
            if not (target / 'stream.json').is_file():
                raise
            existing = json.loads((target / 'receipt.json').read_text())
            if (existing['result']['values'] != receipt['result']['values']
                    or existing['catalog'] != receipt['catalog']
                    or existing.get('import_provenance') != receipt.get('import_provenance')):
                raise ValueError('Conflicting scores or provenance for this evaluation position and protocol')
            return existing
        sync_directory(parent)
    return receipt


def receipt_values(receipt):
    """Yield public metric IDs for legacy observations and named scalar sets."""
    result = receipt.get('result') or {}
    if 'output_ids' in receipt:
        return {f"{receipt['metric_id']}/{key}": result.get('values', {}).get(key)
                for key in receipt['output_ids']}
    return {receipt.get('metric_id') or receipt.get('metric'): result.get('value')}


def import_scores(root, evaluator, *, step, values, outputs, attempt_id, snapshot_identity, snapshot_sha256, protocol):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(evaluator, str) or not METRIC_ID.fullmatch(evaluator):
        raise ValueError('Invalid evaluator ID')
    if type(step) is not int or not 0 <= step < 2**53:
        raise ValueError('Source step must be a nonnegative safe integer')
    if attempt_id is not None and (not isinstance(attempt_id, str) or not 1 <= len(attempt_id) <= 128):
        raise ValueError('Invalid source attempt ID')
    if snapshot_sha256 is not None and (not isinstance(snapshot_sha256, str) or not re.fullmatch('[a-f0-9]{64}', snapshot_sha256)):
        raise ValueError('Invalid snapshot SHA-256')
    if any(v is not None and not isinstance(v, dict) for v in (snapshot_identity, protocol)):
        raise ValueError('Snapshot identity and protocol must be mappings')
    snapshot = finite_json(snapshot_identity or {})
    run_id = manifest['run_id']
    if any(k in snapshot and snapshot[k] != v for k, v in [('run_id', run_id), ('source_step', step)]):
        raise ValueError('Snapshot identity differs from its run or source step')
    if 'attempt_id' in snapshot:
        if attempt_id is not None and snapshot['attempt_id'] != attempt_id:
            raise ValueError('Snapshot identity differs from source attempt')
        attempt_id = snapshot['attempt_id']
    if attempt_id is not None and (not isinstance(attempt_id, str) or not 1 <= len(attempt_id) <= 128):
        raise ValueError('Invalid source attempt ID')
    protocol = finite_json(protocol or {})
    descriptors = scalar_outputs(outputs)
    values = scalar_values(values, descriptors)
    definitions = {}
    for key, descriptor in descriptors.items():
        metric_id = f'{evaluator}/{key}'
        if not METRIC_ID.fullmatch(metric_id):
            raise ValueError(f'Overlong snapshot output ID: {metric_id}')
        definition = dict(descriptor, source='custom:' + evaluator, scope='snapshot',
                          evaluator_id=evaluator, output_id=key)
        definitions[metric_id] = dict(definition, definition_hash=digest(definition))
    revision = write_catalog(root, {'schema_version': 1, 'metrics': definitions})
    identity = {'run_id': run_id, 'attempt_id': attempt_id or 'unknown', 'source_step': step}
    result = {'step': step, 'values': values, 'protocol': protocol, 'snapshot_identity': snapshot}
    receipt = {'schema_version': 1, 'status': 'complete', **identity, 'catalog': revision,
               'output_ids': list(descriptors), 'metric_id': evaluator, 'result': result,
               'import_provenance': snapshot}
    event = {'schema_version': 2, 'event': 'evaluation', 'status': 'complete', 'run_id': run_id,
             'attempt_id': identity['attempt_id'], 'sequence': 1, 'step': step, 'catalog': revision,
             'metrics': {f'{evaluator}/{key}': v for key, v in values.items()}, 'measurement_status': {},
             'snapshot_identity': snapshot, 'source_position_known': True, 'evaluation_protocol': protocol,
             'protocol_sha256': digest(protocol)}
    if snapshot_sha256 is not None:
        receipt['snapshot_sha256'] = event['snapshot_sha256'] = snapshot_sha256
        receipt['import_provenance'] = dict(snapshot_identity=snapshot, snapshot_sha256=snapshot_sha256)
    receipt = publish_scores(root, evaluator, event, receipt)
    from .summary import write_summary_json
    write_summary_json(root)
    return receipt
