"""Full trusted training checkpoints, separate from inference bundles."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from .run_state import atomic_json, sync_directory, validate_event_boundary

SCHEMA = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_rng():
    numpy = np.random.get_state()
    result = {'torch': torch.get_rng_state(), 'python': random.getstate(),
              'numpy': [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]]}
    # CPU execution must not initialize CUDA merely to observe random state.
    if torch.cuda.is_initialized():
        result['cuda'] = torch.cuda.get_rng_state_all()
        result['cuda_device'] = torch.cuda.current_device()
    return result


def restore_rng(state):
    if 'cuda' in state:
        if (not torch.cuda.is_available() or type(state['cuda']) is not list
                or len(state['cuda']) != torch.cuda.device_count()
                or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != 'cpu' for value in state['cuda'])
                or type(state.get('cuda_device')) is not int or not 0 <= state['cuda_device'] < torch.cuda.device_count()):
            raise ValueError('Checkpoint CUDA RNG device inventory differs from the runtime')
        torch.cuda.set_rng_state_all(state['cuda'])
        torch.cuda.set_device(state['cuda_device'])
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))


def data_contract(data, specification):
    has_save, has_load = callable(getattr(data, 'state_dict', None)), callable(getattr(data, 'load_state_dict', None))
    if has_save != has_load:
        raise ValueError('Data recovery requires both state_dict and load_state_dict')
    builtin = specification['factory'] in ('gaussian_grid', 'paired_linear')
    stateless = builtin or getattr(data, 'resume_stateless', False) is True
    identity_fn = getattr(data, 'resume_identity', None)
    identity = identity_fn() if callable(identity_fn) else {'specification': specification}
    # Require a stable portable descriptor, not repr(object) or pickle identity.
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    return {'supported': bool(has_save or stateless), 'stateful': has_save, 'identity': identity}


def _portable(value):
    if value is None or type(value) in (bool, int, float, str) or isinstance(value, torch.Tensor):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _portable(item)
        return
    if isinstance(value, dict) and all(type(key) in (str, int) for key in value):
        for item in value.values():
            _portable(item)
        return
    raise ValueError(f'Checkpoint state must contain tensors and primitive containers, got {type(value).__name__}')


def trainer_state(trainer, last_batch):
    rng = capture_rng()
    try:
        modules = ('graph', 'prior', 'ema_graph', 'ema_prior')
        state = {name: getattr(trainer, name).state_dict() for name in modules}
        state.update(optimizers=[trainer.opt_g.state_dict(), trainer.opt_d.state_dict()],
                     base_lrs=trainer.base_lrs, step=trainer.step,
                     streams={name: stream.get_state() for name, stream in trainer.streams.items()},
                     rng=rng, data=trainer.data.state_dict() if callable(getattr(trainer.data, 'state_dict', None)) else None,
                     buffers={name: dict(getattr(trainer, name).named_buffers()) for name in modules},
                     trainable={name: {key: value.requires_grad for key, value in getattr(trainer, name).named_parameters()} for name in modules},
                     modes={name: {key: module.training for key, module in getattr(trainer, name).named_modules()} for name in modules},
                     last_batch=last_batch)
        _portable(state)
        return state
    finally:
        restore_rng(rng)


def _restore_trainer(trainer, state, metadata=None):
    required = {'graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs',
                'step', 'streams', 'rng', 'data', 'modes', 'buffers', 'trainable', 'last_batch'}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError('Checkpoint training state fields are incomplete or unsupported')
    if getattr(trainer, 'device', torch.device('cpu')).type == 'cuda':
        rng = state['rng']
        if not isinstance(rng, dict) or 'cuda' not in rng or 'cuda_device' not in rng:
            raise ValueError('CUDA training checkpoint is missing complete CUDA RNG state')
        if type(rng['cuda']) is not list or len(rng['cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG device inventory differs from the runtime')
        for index, saved in enumerate(rng['cuda']):
            if (not isinstance(saved, torch.Tensor) or saved.dtype != torch.uint8 or saved.device.type != 'cpu'
                    or saved.shape != torch.cuda.get_rng_state(index).shape):
                raise ValueError('Checkpoint CUDA RNG tensor is incompatible')
    if type(state['step']) is not int or not 0 <= state['step'] <= trainer.config['training']['steps']:
        raise ValueError('Checkpoint step is outside the configured training schedule')
    if len(state['optimizers']) != 2 or len(state['base_lrs']) != 2:
        raise ValueError('Checkpoint requires both optimizer states and base learning rates')
    if set(state['streams']) != set(trainer.streams):
        raise ValueError('Checkpoint named RNG streams differ from trainer')
    if set(state['modes']) != {'graph', 'prior', 'ema_graph', 'ema_prior'}:
        raise ValueError('Checkpoint module mode inventory differs from trainer')
    for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
        module = getattr(trainer, name)
        if set(state[name]) != set(module.state_dict()):
            raise ValueError('Checkpoint model state inventory differs from trainer')
        for key, value in module.state_dict().items():
            if isinstance(value, torch.Tensor) and (not isinstance(state[name][key], torch.Tensor) or state[name][key].shape != value.shape or state[name][key].dtype != value.dtype):
                raise ValueError('Checkpoint model tensor shape or dtype differs from trainer')
        if set(state['buffers'][name]) != dict(module.named_buffers()).keys() or set(state['trainable'][name]) != dict(module.named_parameters()).keys():
            raise ValueError('Checkpoint buffer/parameter inventory differs from trainer')
        for key, value in module.named_buffers():
            saved = state['buffers'][name][key]
            if saved.shape != value.shape or saved.dtype != value.dtype:
                raise ValueError('Checkpoint buffer shape or dtype differs from trainer')
        if set(state['modes'][name]) != dict(getattr(trainer, name).named_modules()).keys():
            raise ValueError('Checkpoint nested module mode inventory differs from trainer')
    for optimizer, saved, rates in zip((trainer.opt_g, trainer.opt_d), state['optimizers'], state['base_lrs']):
        if set(saved) != {'state', 'param_groups'} or len(saved['param_groups']) != len(optimizer.param_groups) or len(rates) != len(optimizer.param_groups):
            raise ValueError('Checkpoint optimizer parameter groups differ from trainer')
        for actual, prior in zip(optimizer.param_groups, saved['param_groups']):
            if len(actual['params']) != len(prior['params']):
                raise ValueError('Checkpoint optimizer parameter inventory differs from trainer')
    from .tuning_overrides import checkpoint_base_lrs
    expected_rates, overridden = checkpoint_base_lrs(trainer, metadata)
    if state['base_lrs'] != expected_rates:
        raise ValueError('Checkpoint base learning rates differ from original configuration or validated startup override')
    if overridden:
        from particlegan import learning_rate_scale
        settings = trainer.config['training']
        scale = (learning_rate_scale(state['step'] - 1, settings['steps'],
                                    start=settings['lr_anneal_start'], floor=settings['lr_floor'])
                 if state['step'] else 1.0)
        for saved, rates in zip(state['optimizers'], expected_rates):
            for group, rate in zip(saved['param_groups'], rates):
                if type(group.get('lr')) not in (int, float) or group['lr'] != rate * scale:
                    raise ValueError('Checkpoint optimizer learning rate differs from validated startup schedule')
    for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
        module = getattr(trainer, name)
        module.load_state_dict(state[name], strict=True)
        with torch.no_grad():
            for key, buffer in module.named_buffers():
                buffer.copy_(state['buffers'][name][key])
        for key, parameter in module.named_parameters():
            parameter.requires_grad_(state['trainable'][name][key])
        for key, child in module.named_modules():
            child.training = state['modes'][name][key]
    for optimizer, saved in zip((trainer.opt_g, trainer.opt_d), state['optimizers']):
        optimizer.load_state_dict(saved)
    trainer.base_lrs = state['base_lrs']
    trainer.step = state['step']
    for name, stream in trainer.streams.items():
        stream.set_state(state['streams'][name])
    if state['data'] is not None:
        trainer.data.load_state_dict(state['data'])
    # Constructors and custom load hooks can consume global randomness.
    restore_rng(state['rng'])
    from .recipes import move_tensors
    return move_tensors(state['last_batch'], getattr(trainer, 'device', 'cpu'))


def restore_trainer(trainer, state, *, metadata=None):
    try:
        return _restore_trainer(trainer, state, metadata)
    except (KeyError, TypeError, AttributeError, IndexError, RuntimeError) as exc:
        raise ValueError(f'Invalid training checkpoint state: {exc}') from exc


def write_checkpoint(run_dir, trainer, last_batch, metadata):
    root = Path(run_dir) / 'checkpoints'
    root.mkdir(exist_ok=True)
    sync_directory(root.parent)
    name = f"{metadata['attempt_id']}-step-{trainer.step:08d}-{uuid.uuid4().hex[:12]}"
    temporary, target = root / ('.pending-' + name), root / name
    temporary.mkdir()
    try:
        state = trainer_state(trainer, last_batch)
        with (temporary / 'state.pt').open('wb') as output:
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        digest = file_sha256(temporary / 'state.pt')
        info = dict(metadata, schema_version=SCHEMA, kind='hypergan-training-checkpoint', step=trainer.step, state_sha256=digest)
        atomic_json(temporary / 'manifest.json', info)
        sync_directory(temporary)
        temporary.rename(target)
        sync_directory(root)
        atomic_json(root / 'latest.json', {'schema_version': SCHEMA, 'checkpoint': target.name, 'step': trainer.step})
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_checkpoint(run_dir, checkpoint=None):
    root = (Path(run_dir) / 'checkpoints').resolve()
    if checkpoint is None:
        pointer = root / 'latest.json'
        if not pointer.exists():
            raise ValueError('No full training checkpoint exists; an inference model is not resumable')
        latest = json.loads(pointer.read_text())
        if not isinstance(latest, dict) or latest.get('schema_version') != SCHEMA or not isinstance(latest.get('checkpoint'), str):
            raise ValueError('Invalid training checkpoint pointer')
        checkpoint = root / latest['checkpoint']
    target = Path(checkpoint)
    if not target.is_absolute():
        choices = (target, Path(run_dir) / target, root / target)
        target = next((choice for choice in choices if choice.is_dir()), root / target)
    target = target.resolve()
    if not target.is_relative_to(root) or not target.is_dir() or target.name.startswith('.'):
        raise ValueError('Checkpoint must be a completed checkpoint directory inside this run')
    info = json.loads((target / 'manifest.json').read_text())
    required = {'schema_version', 'kind', 'run_id', 'attempt_id', 'step', 'config', 'config_sha256', 'runtime', 'data_contract', 'implementation', 'state_sha256', 'next_sample_sequence'}
    if not isinstance(info, dict) or not required.issubset(info):
        raise ValueError('Invalid training checkpoint metadata')
    if info.get('schema_version') != SCHEMA or info.get('kind') != 'hypergan-training-checkpoint':
        raise ValueError('Unsupported full training checkpoint schema')
    from .checkpoint_compatibility import validate_checkpoint_compatibility
    validate_checkpoint_compatibility(info)
    validate_event_boundary(run_dir, info)
    if file_sha256(target / 'state.pt') != info['state_sha256']:
        raise ValueError('Training checkpoint digest mismatch')
    try:
        state = torch.load(target / 'state.pt', map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        raise ValueError('Invalid training checkpoint payload; safe tensor loading failed') from exc
    if not isinstance(state, dict) or 'step' not in state:
        raise ValueError('Invalid training checkpoint state fields')
    if state['step'] != info['step']:
        raise ValueError('Checkpoint step metadata mismatch')
    return target, info, state


def read_checkpoint(run_dir, checkpoint=None):
    try:
        return _read_checkpoint(run_dir, checkpoint)
    except (KeyError, TypeError, OSError, EOFError) as exc:
        raise ValueError(f'Invalid or incomplete training checkpoint: {exc}') from exc
