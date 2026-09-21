"""Torch-free, one-at-a-time preview rendering with broker-owned process cleanup."""
import hashlib
import json
import os
from pathlib import Path

from .cpu_worker_service import CPUWorkerService
from .previews import DEFAULT_KEEP, MAX_RENDER_BYTES


def _factory(*args):
    # Rendering is observation and must never contend for a training device.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    if hasattr(os, 'nice'):
        os.nice(10)
    from .preview_snapshot import renderer_factory
    return renderer_factory(*args)


def _handler(*args):
    from .preview_snapshot import renderer_command
    state, operation, payload = args
    if operation != 'render-publish':
        return renderer_command(*args)
    from .previews import publish_preview_payload, _write_bounded
    receipt = renderer_command(state, 'render', None)
    path, descriptor, identity, step, output = state
    rendered = _read_output(output, receipt, identity, step)
    record, index, errors = publish_preview_payload(payload['run_dir'], rendered, identity, step, payload['keep'])
    result = Path(output).with_name('publication.json')
    size = _write_bounded(result, {'record': record, 'index': index, 'errors': errors})
    return {'bytes': size, 'sha256': hashlib.sha256(result.read_bytes()).hexdigest(),
            'step': step, 'identity': identity}


def render_snapshot(path, descriptor, identity, step, output, *, timeout=60, publish_run_dir=None,
                    keep=DEFAULT_KEEP, cancellation_event=None):
    """Finish or reap the isolated renderer before returning bounded JSON.

    PreviewWorker runs this blocking operation on its supervising thread.
    The broker monitors coordinator death independently of renderer Python/native
    code. Deadline cleanup has the same finite grace as CPUWorkerService.
    """
    service = CPUWorkerService(_factory, _handler,
        args=(str(path), descriptor, identity, step, str(output)),
        run_id=identity['run_id'], attempt_id=identity['attempt_id'], world_size=1,
        initialize_process_group=False, startup_timeout=timeout, command_timeout=timeout,
        collective_timeout=timeout, total_timeout=timeout, cancellation_event=cancellation_event)
    with service:
        response = service.command('render' if publish_run_dir is None else 'render-publish',
            None if publish_run_dir is None else {'run_dir': str(publish_run_dir), 'keep': keep})
        receipt = response['results'][0]
    if publish_run_dir is not None:
        output = Path(output).with_name('publication.json')
    return _read_output(output, receipt, identity, step)


def _read_output(output, receipt, identity, step):
    output = Path(output)
    if output.is_symlink() or not output.is_file() or not 0 < output.stat().st_size <= MAX_RENDER_BYTES:
        raise ValueError('Preview renderer output exceeds its bounded ordinary-file contract')
    with output.open('rb') as source:
        encoded = source.read(MAX_RENDER_BYTES + 1)
    if (len(encoded) > MAX_RENDER_BYTES or receipt['bytes'] != len(encoded)
            or receipt['sha256'] != hashlib.sha256(encoded).hexdigest()
            or receipt['step'] != step or receipt['identity'] != identity):
        raise ValueError('Preview renderer output size, hash or identity mismatch')
    return json.loads(encoded)
