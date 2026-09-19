"""Torch-free, one-at-a-time preview rendering with broker-owned process cleanup."""
import hashlib
import json
from pathlib import Path

from .cpu_worker_service import CPUWorkerService
from .previews import MAX_BYTES


def _factory(*args):
    from .preview_snapshot import renderer_factory
    return renderer_factory(*args)


def _handler(*args):
    from .preview_snapshot import renderer_command
    return renderer_command(*args)


def render_snapshot(path, descriptor, identity, step, output, *, timeout=60):
    """Finish or reap the isolated renderer before returning bounded JSON.

    The caller submits synchronously, so there is at most one pending preview.
    The broker monitors coordinator death independently of renderer Python/native
    code. Deadline cleanup has the same finite grace as CPUWorkerService.
    """
    service = CPUWorkerService(_factory, _handler,
        args=(str(path), descriptor, identity, step, str(output)),
        run_id=identity['run_id'], attempt_id=identity['attempt_id'], world_size=1,
        initialize_process_group=False, startup_timeout=timeout, command_timeout=timeout,
        collective_timeout=timeout, total_timeout=timeout)
    with service:
        response = service.command('render')
        receipt = response['results'][0]
    output = Path(output)
    if output.is_symlink() or not output.is_file() or not 0 < output.stat().st_size <= MAX_BYTES:
        raise ValueError('Preview renderer output exceeds its bounded ordinary-file contract')
    with output.open('rb') as source:
        encoded = source.read(MAX_BYTES + 1)
    if (len(encoded) > MAX_BYTES or receipt['bytes'] != len(encoded)
            or receipt['sha256'] != hashlib.sha256(encoded).hexdigest()
            or receipt['step'] != step or receipt['identity'] != identity):
        raise ValueError('Preview renderer output size, hash or identity mismatch')
    return json.loads(encoded)
