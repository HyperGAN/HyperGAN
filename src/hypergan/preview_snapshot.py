"""Worker-only immutable preview capture and group-free numerical rendering.

Custom serialization hooks run on copied modules. Constructors, deepcopy hooks
and forwards remain trusted Python: the supervisor bounds their execution time,
not arbitrary allocations, hidden external state or subprocesses they create.
"""
import copy
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
import torch.distributed as dist

from .artifacts import _restore_buffers
from .checkpoints import _portable, capture_rng, restore_rng
from .previews import MAX_COUNT, MAX_ELEMENTS, _inputs, _write_bounded, render_preview
from .recipes import ComponentGraph, make_prior, move_tensors

MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


class _BoundedWriter:
    def __init__(self, stream):
        self.stream, self.size, self.error = stream, 0, None

    def write(self, data):
        try:
            if self.size + (data.nbytes if isinstance(data, memoryview) else len(data)) > MAX_SNAPSHOT_BYTES:
                raise ValueError(f'Preview snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes')
            count = self.stream.write(data)
            self.size += count
            return count
        except BaseException as error:
            if self.error is None:
                self.error = error
            raise

    def flush(self):
        self.stream.flush()


def capture_snapshot(trainer, batch, identity, path):
    """Write only copied EMA/prior state and bounded conditioning at a boundary."""
    rng, threads = capture_rng(), torch.get_num_threads()
    streams = {name: stream.get_state() for name, stream in trainer.streams.items()}
    try:
        inputs = _inputs(trainer, batch)
        real = batch['real']
        if not isinstance(real, torch.Tensor) or real.ndim < 1 or not len(real):
            raise ValueError('Preview capture requires a nonempty completed real batch')
        elements = real[0].numel()
        for name, value in inputs.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or not len(value):
                raise ValueError(f'Preview input {name} must be a nonempty batched tensor')
            elements += value[0].numel()
        count = min(trainer.config['sampling']['count'], MAX_COUNT, MAX_ELEMENTS // max(1, elements))
        if count < 1:
            raise ValueError(f'One preview sample exceeds the {MAX_ELEMENTS}-element budget')
        normalized = {name: value[torch.arange(count) % len(value)].detach().clone() for name, value in inputs.items()}
        # A single real row supplies output-shape budgeting only; it is not rendered.
        normalized['real'] = real[:1].detach().clone() if 'real' not in normalized else normalized['real']
        needed = set()
        def include(name):
            if name in needed:
                return
            needed.add(name)
            for binding in trainer.config['components'][name]['inputs'].values():
                if binding.startswith('components.'):
                    include(binding.split('.')[1])
        include('generator')
        memo = {id(module): module for module in list(sys.modules.values()) if isinstance(module, ModuleType)}
        models = {name: copy.deepcopy(trainer.ema_graph.models[name], memo) for name in needed}
        prior = copy.deepcopy(trainer.ema_prior, memo)
        config = copy.deepcopy(trainer.config)
        config['components'] = {name: spec for name, spec in config['components'].items() if name in needed}
        state = {'schema_version': 1, 'kind': 'hypergan-preview-snapshot', 'step': trainer.step,
                 'identity': copy.deepcopy(identity), 'config': config,
                 'model_states': {name: model.state_dict() for name, model in models.items()},
                 'model_buffers': {name: dict(model.named_buffers()) for name, model in models.items()},
                 'prior': prior.state_dict(), 'prior_buffers': dict(prior.named_buffers()), 'batch': normalized}
        _portable(state)
        state = move_tensors(state, 'cpu')
        path = Path(path)
        with path.open('xb') as output:
            writer = _BoundedWriter(output)
            try:
                torch.save(state, writer)
            except BaseException as error:
                # Torch's ZIP finalizer can replace an original failed write
                # with an unrelated offset error; retain the actual byte/I/O cause.
                if writer.error is not None:
                    raise writer.error from error
                raise
            output.flush()
            os.fsync(output.fileno())
        return {'bytes': path.stat().st_size, 'sha256': _sha256(path)}
    finally:
        restore_rng(rng)
        for name, value in streams.items():
            trainer.streams[name].set_state(value)
        torch.set_num_threads(threads)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def renderer_factory(rank, world_size, path, descriptor, identity, step, output):
    if dist.is_initialized():
        raise ValueError('Preview renderer must not have a process group')
    torch.set_num_threads(1)
    return path, descriptor, identity, step, output


def renderer_command(state, operation, payload):
    if operation != 'render':
        raise ValueError('Unknown preview renderer operation')
    path, descriptor, identity, step, output = state
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_SNAPSHOT_BYTES:
        raise ValueError('Invalid bounded preview snapshot file')
    if path.stat().st_size != descriptor['bytes'] or _sha256(path) != descriptor['sha256']:
        raise ValueError('Preview snapshot size or hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('schema_version') != 1 or saved.get('kind') != 'hypergan-preview-snapshot'
            or saved.get('step') != step or saved.get('identity') != identity):
        raise ValueError('Preview snapshot identity or step differs from requested render')
    # Set every constructor's RNG explicitly, not just the later forward RNG.
    import random
    import numpy as np
    seed = saved['config']['sampling']['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    graph = ComponentGraph(saved['config']['components']).float()
    for name, model in graph.models.items():
        model.load_state_dict(saved['model_states'][name])
        _restore_buffers(model, saved['model_buffers'][name])
    prior = make_prior(saved['config']['prior'], device='cpu').float()
    prior.load_state_dict(saved['prior'])
    _restore_buffers(prior, saved['prior_buffers'])
    trainer = SimpleNamespace(config=saved['config'], step=step, ema_graph=graph, ema_prior=prior)
    result = render_preview(trainer, saved['batch'], identity)
    size = _write_bounded(Path(output), result)
    return {'bytes': size, 'sha256': _sha256(output), 'step': step, 'identity': identity}
