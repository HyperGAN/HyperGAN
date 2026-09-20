"""Worker-only immutable preview capture and group-free numerical rendering.

Custom serialization hooks run on copied modules. Constructors, deepcopy hooks
and forwards remain trusted Python: the supervisor bounds their execution time,
not arbitrary allocations, hidden external state or subprocesses they create.
"""
import copy
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import torch
import torch.distributed as dist

from .artifacts import _restore_buffers
from .checkpoints import _portable, capture_rng, restore_rng
from .previews import MAX_COUNT, MAX_ELEMENTS, _inputs, _write_bounded, render_preview
from .recipes import ComponentGraph, make_prior

MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


class _BoundedWriter:
    def __init__(self, stream, check=None):
        self.stream, self.size, self.error = stream, 0, None
        self.check = check or (lambda: None)

    def write(self, data):
        try:
            self.check()
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
        self.check()
        self.stream.flush()


def capture_snapshot_state(trainer, batch, identity):
    """Freeze owned CPU state at a boundary; perform no filesystem operations.

    Custom copy/state hooks and device transfers finish under the training RNG
    fence. The returned snapshot contains no live modules or tensor aliases.
    """
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
        needed.update(trainer.config['components'][name]['reuse'] for name in list(needed)
                      if 'reuse' in trainer.config['components'][name])
        models = {name: copy.deepcopy(trainer.ema_graph.models[name], memo) for name in needed
                  if 'reuse' not in trainer.config['components'][name]}
        prior = copy.deepcopy(trainer.ema_prior, memo)
        config = copy.deepcopy(trainer.config)
        config['components'] = {name: spec for name, spec in config['components'].items() if name in needed}
        state = {'schema_version': 1, 'kind': 'hypergan-preview-snapshot', 'step': trainer.step,
                 'identity': copy.deepcopy(identity), 'config': config,
                 'model_states': {name: model.state_dict() for name, model in models.items()},
                 'model_buffers': {name: dict(model.named_buffers()) for name, model in models.items()},
                 'prior': prior.state_dict(), 'prior_buffers': dict(prior.named_buffers()), 'batch': normalized}
        _portable(state)
        return _freeze_cpu_state(state)
    finally:
        restore_rng(rng)
        for name, value in streams.items():
            trainer.streams[name].set_state(value)
        torch.set_num_threads(threads)


def _freeze_cpu_state(state):
    """Detach data from custom containers/tensor reducers before background save."""
    budget = [MAX_SNAPSHOT_BYTES, 1000000]
    memo = {}
    def freeze(value, depth=0):
        budget[1] -= 1
        if depth > 64 or budget[1] < 0:
            raise ValueError('Preview snapshot exceeds its structure budget')
        if isinstance(value, torch.Tensor):
            if id(value) in memo:
                return memo[id(value)]
            budget[0] -= value.numel() * value.element_size()
            if budget[0] < 0:
                raise ValueError(f'Preview snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes')
            tensor = value.detach()
            if type(tensor) is not torch.Tensor:
                tensor = tensor.as_subclass(torch.Tensor)
            tensor = tensor.to(device='cpu', copy=True)
            memo[id(value)] = tensor
            return tensor
        if value is None or type(value) in (bool, int, float, str):
            budget[0] -= len(value) * 12 + 2 if type(value) is str else 32
            result = value
        elif isinstance(value, dict):
            # Preserve standard state_dict version metadata without retaining
            # custom mapping classes or user-defined pickle hooks.
            result = OrderedDict() if isinstance(value, OrderedDict) else {}
            for key, child in value.items():
                if type(key) not in (str, int):
                    raise ValueError('Preview state keys must be strings or integers')
                result[freeze(key, depth + 1)] = freeze(child, depth + 1)
            if isinstance(value, OrderedDict) and hasattr(value, '_metadata'):
                result._metadata = freeze(value._metadata, depth + 1)
        elif isinstance(value, (list, tuple)):
            result = [freeze(child, depth + 1) for child in value]
            if isinstance(value, tuple):
                result = tuple(result)
        else:
            raise ValueError(f'Preview state must contain tensors and primitives, got {type(value).__name__}')
        if budget[0] < 0:
            raise ValueError(f'Preview snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes')
        return result
    return freeze(state)


def write_snapshot(state, path, *, cancellation_event=None, deadline=None):
    """Persist an owned CPU snapshot without accessing a trainer or RNG state."""
    def check():
        if cancellation_event is not None and cancellation_event.is_set():
            raise RuntimeError('Preview cancelled during snapshot persistence')
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Preview snapshot persistence deadline exceeded')
    check()
    path = Path(path)
    with path.open('xb') as output:
        writer = _BoundedWriter(output, check)
        try:
            torch.save(state, writer)
        except BaseException as error:
            # Torch's ZIP finalizer may mask the original bounded write error.
            if writer.error is not None:
                raise writer.error from error
            raise
        check()
        output.flush()
        os.fsync(output.fileno())
        check()
    return {'bytes': path.stat().st_size, 'sha256': _sha256(path, check=check)}


def capture_snapshot(trainer, batch, identity, path):
    """Synchronous file handoff for replicated ranks and direct snapshot callers."""
    return write_snapshot(capture_snapshot_state(trainer, batch, identity), path)


def _sha256(path, *, check=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            if check is not None:
                check()
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
