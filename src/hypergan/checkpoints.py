"""Full trusted training checkpoints, separate from inference bundles."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import re
import shutil
import uuid

from ._file_lock import lock_file, retry_lock, unlock_file
from .run_state import atomic_json, run_lock, sync_directory, validate_event_boundary

SCHEMA = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_rng():
    import numpy as np
    import torch
    numpy = np.random.get_state()
    result = {'torch': torch.get_rng_state(), 'python': random.getstate(),
              'numpy': [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]]}
    # CPU execution must not initialize CUDA merely to observe random state.
    if torch.cuda.is_initialized():
        result['cuda'] = torch.cuda.get_rng_state_all()
        result['cuda_device'] = torch.cuda.current_device()
    return result


def restore_rng(state):
    import numpy as np
    import torch
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
    import torch
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


def _restore_trainer(trainer, state):
    import numpy as np
    import torch
    required = {'graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs',
                'step', 'streams', 'rng', 'data', 'modes', 'buffers', 'trainable', 'last_batch'}
    if not isinstance(state, dict) or not required <= set(state) <= required | {'aggregation'}:
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
    # Validate consumed RNG streams before any model or optimizer is restored.
    # Local generators leave the process-global streams untouched on failure.
    for name, stream in trainer.streams.items():
        torch.Generator(device=stream.device).set_state(state['streams'][name])
    torch.Generator(device='cpu').set_state(state['rng']['torch'])
    random.Random().setstate(state['rng']['python'])
    numpy_state = state['rng']['numpy']
    np.random.RandomState().set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
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
        if any(type(value) is not bool for value in (*state['modes'][name].values(), *state['trainable'][name].values())):
            raise ValueError('Checkpoint module modes and trainability must be boolean')
    for optimizer, saved, rates in zip((trainer.opt_g, trainer.opt_d), state['optimizers'], state['base_lrs']):
        # The recipe's optimizers keep their regularization state (EMA critic,
        # LR record, latent-damping history) under 'regularizer'.
        if set(saved) != set(optimizer.state_dict()) or len(saved['param_groups']) != len(optimizer.param_groups) or len(rates) != len(optimizer.param_groups):
            raise ValueError('Checkpoint optimizer parameter groups differ from trainer')
        for actual, prior in zip(optimizer.param_groups, saved['param_groups']):
            if len(actual['params']) != len(prior['params']):
                raise ValueError('Checkpoint optimizer parameter inventory differs from trainer')
        validate = getattr(optimizer, 'validate_state_dict', None)
        if validate is not None:
            validate(saved)
    if state['base_lrs'] != trainer.base_lrs:
        raise ValueError('Checkpoint base learning rates differ from original configuration')
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
    if 'aggregation' in state:
        strategy = getattr(trainer, 'aggregation_strategy', getattr(trainer, 'aggregation', None))
        if strategy is not None and hasattr(strategy, 'load_state_dict') and callable(strategy.load_state_dict):
            strategy.load_state_dict(state['aggregation'])
    # Constructors and custom load hooks can consume global randomness.
    restore_rng(state['rng'])
    from .recipes import move_tensors
    return move_tensors(state['last_batch'], getattr(trainer, 'device', 'cpu'))


def restore_trainer(trainer, state):
    try:
        return _restore_trainer(trainer, state)
    except (KeyError, TypeError, AttributeError, IndexError, RuntimeError) as exc:
        raise ValueError(f'Invalid training checkpoint state: {exc}') from exc


def _adapt_last_batch(last_batch, target_batch_size, device, *, rank=0):
    import torch
    from .recipes import move_tensors
    if last_batch is None:
        return None

    def adapt(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or target_batch_size is None or target_batch_size <= 0:
                return value.detach().clone().to(device)
            n = value.shape[0]
            if n == target_batch_size:
                return value.detach().clone().to(device)
            start = rank * target_batch_size
            if n >= start + target_batch_size:
                return value[start:start + target_batch_size].detach().clone().to(device)
            if n > 0:
                repeats = (target_batch_size + n - 1) // n
                tiled = value.repeat(repeats, *([1] * (value.ndim - 1)))
                return tiled[:target_batch_size].detach().clone().to(device)
            return value.detach().clone().to(device)
        if isinstance(value, dict):
            return {k: adapt(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(adapt(v) for v in value)
        return move_tensors(value, device)

    return adapt(last_batch)


def _restore_trainer_cross_topology(trainer, state, *, rank=0, world_size=1):
    import copy
    import numpy as np
    import torch
    from .recipes import move_tensors
    from .training import _skip_cpu_capture_guard

    required = {'graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs',
                'step', 'streams', 'rng', 'data', 'modes', 'buffers', 'trainable', 'last_batch'}
    if not isinstance(state, dict) or not required <= set(state) <= required | {'aggregation'}:
        raise ValueError('Checkpoint training state fields are incomplete or unsupported')
    if type(state['step']) is not int or not 0 <= state['step'] <= trainer.config['training']['steps']:
        raise ValueError('Checkpoint step is outside the configured training schedule')
    if len(state['optimizers']) != 2 or len(state['base_lrs']) != 2:
        raise ValueError('Checkpoint requires both optimizer states and base learning rates')
    if not isinstance(state['streams'], dict):
        raise ValueError('Checkpoint named RNG streams must be a dictionary')
    if set(state['modes']) != {'graph', 'prior', 'ema_graph', 'ema_prior'}:
        raise ValueError('Checkpoint module mode inventory differs from trainer')
    for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
        module = getattr(trainer, name)
        if set(state[name]) != set(module.state_dict()):
            raise ValueError('Checkpoint model state inventory differs from trainer')
        for key, value in module.state_dict().items():
            if isinstance(value, torch.Tensor) and (
                not isinstance(state[name][key], torch.Tensor)
                or state[name][key].shape != value.shape
                or state[name][key].dtype != value.dtype
            ):
                raise ValueError('Checkpoint model tensor shape or dtype differs from trainer')
        if (
            set(state['buffers'][name]) != dict(module.named_buffers()).keys()
            or set(state['trainable'][name]) != dict(module.named_parameters()).keys()
        ):
            raise ValueError('Checkpoint buffer/parameter inventory differs from trainer')
        for key, value in module.named_buffers():
            saved_buf = state['buffers'][name][key]
            if saved_buf.shape != value.shape or saved_buf.dtype != value.dtype:
                raise ValueError('Checkpoint buffer shape or dtype differs from trainer')
        if set(state['modes'][name]) != dict(getattr(trainer, name).named_modules()).keys():
            raise ValueError('Checkpoint nested module mode inventory differs from trainer')
        if any(type(value) is not bool for value in (*state['modes'][name].values(), *state['trainable'][name].values())):
            raise ValueError('Checkpoint module modes and trainability must be boolean')

    target_device = torch.device(getattr(trainer, 'device', 'cpu'))
    prepared_optimizers = []
    for optimizer, saved, rates in zip((trainer.opt_g, trainer.opt_d), state['optimizers'], state['base_lrs']):
        if (
            set(saved) != set(optimizer.state_dict())
            or len(saved['param_groups']) != len(optimizer.param_groups)
            or len(rates) != len(optimizer.param_groups)
        ):
            raise ValueError('Checkpoint optimizer parameter groups differ from trainer')
        saved_copy = copy.deepcopy(saved)
        for actual, prior in zip(optimizer.param_groups, saved_copy['param_groups']):
            if len(actual['params']) != len(prior['params']):
                raise ValueError('Checkpoint optimizer parameter inventory differs from trainer')
            for dev_flag in ('fused', 'capturable'):
                if dev_flag in actual:
                    prior[dev_flag] = actual[dev_flag]
                elif target_device.type == 'cpu' and prior.get(dev_flag):
                    prior[dev_flag] = False
        if 'regularizer' in saved_copy:
            saved_copy['regularizer'] = move_tensors(saved_copy['regularizer'], target_device)
        validate = getattr(optimizer, 'validate_state_dict', None)
        if validate is not None:
            validate(saved_copy)
        prepared_optimizers.append(saved_copy)
    if state['base_lrs'] != trainer.base_lrs:
        raise ValueError('Checkpoint base learning rates differ from original configuration')

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

    for optimizer, saved_copy in zip((trainer.opt_g, trainer.opt_d), prepared_optimizers):
        optimizer.load_state_dict(saved_copy)
        for group in optimizer.param_groups:
            use_device_step = bool(group.get('fused') or group.get('capturable')) and target_device.type != 'cpu'
            step_device = target_device if use_device_step else torch.device('cpu')
            for p in group['params']:
                p_state = optimizer.state.get(p)
                if isinstance(p_state, dict) and isinstance(p_state.get('step'), torch.Tensor):
                    p_state['step'] = p_state['step'].to(step_device)
        if target_device.type == 'cpu':
            _skip_cpu_capture_guard(optimizer)

    trainer.base_lrs = copy.deepcopy(state['base_lrs'])
    trainer.step = state['step']

    settings = trainer.config['training']
    base_seed = int(settings['seed'])
    data_offset = int(settings.get('data_seed_offset', 1))
    prior_offset = int(settings.get('prior_seed_offset', 2))
    step_val = int(state['step'])
    rank_mix = (base_seed + (step_val + 1) * 2654435761 + rank * 104729 + world_size * 65537) % (2 ** 63)
    offsets = {'prior': prior_offset, 'penalty': 3, 'noise': 5}

    for name, stream in trainer.streams.items():
        if name == 'data':
            saved_data_stream = state['streams'].get('data')
            restored_data_stream = False
            if (
                isinstance(saved_data_stream, torch.Tensor)
                and saved_data_stream.dtype == torch.uint8
                and saved_data_stream.shape == stream.get_state().shape
            ):
                try:
                    torch.Generator(device=stream.device).set_state(saved_data_stream)
                    stream.set_state(saved_data_stream)
                    restored_data_stream = True
                except RuntimeError:
                    restored_data_stream = False
            if not restored_data_stream:
                stream.manual_seed((base_seed + data_offset + 1000003 * (step_val + 1)) % (2 ** 63))
        else:
            stream.manual_seed((rank_mix + offsets.get(name, 17)) % (2 ** 63))

    if state['data'] is not None and callable(getattr(trainer.data, 'load_state_dict', None)):
        saved_data_state = copy.deepcopy(state['data'])
        if not (isinstance(saved_data_state, dict) and 'owner_rank' in saved_data_state and saved_data_state['owner_rank'] != rank):
            trainer.data.load_state_dict(saved_data_state)

    strategy = getattr(trainer, 'aggregation_strategy', getattr(trainer, 'aggregation', None))
    if strategy is not None:
        saved_agg = state.get('aggregation')
        if (
            isinstance(saved_agg, dict)
            and saved_agg.get('strategy') == getattr(strategy, 'name', None)
            and callable(getattr(strategy, 'load_state_dict', None))
        ):
            strategy.load_state_dict(move_tensors(copy.deepcopy(saved_agg), 'cpu'))
        else:
            if callable(getattr(strategy, 'initialize', None)):
                strategy.initialize()
            if hasattr(strategy, 'step_count'):
                strategy.step_count = step_val
            if hasattr(strategy, 'last_synced_step'):
                strategy.last_synced_step = step_val

    torch.manual_seed(rank_mix)
    random.seed(rank_mix)
    np.random.seed(rank_mix % (2 ** 32))

    target_bs = getattr(trainer, 'local_batch_size', settings['batch_size'])
    return _adapt_last_batch(state['last_batch'], target_bs, target_device, rank=rank)


def restore_trainer_cross_topology(trainer, state, *, rank=0, world_size=1):
    try:
        return _restore_trainer_cross_topology(trainer, state, rank=rank, world_size=world_size)
    except (KeyError, TypeError, AttributeError, IndexError, RuntimeError) as exc:
        raise ValueError(f'Invalid training checkpoint state for cross-topology restore: {exc}') from exc


class _HashingWriter:
    """Digest bytes as torch.save streams them, so the payload is never re-read."""
    def __init__(self, output):
        self.output, self.digest = output, hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.output.write(data)

    def flush(self):
        self.output.flush()


_PENDING_CHECKPOINT = re.compile(r'\.pending-.+-step-\d+-[0-9a-f]{12}')


def _sweep_pending(root):
    """Remove pending checkpoint writes that a killed writer never cleaned up.

    The caller is the run-lock owner, so no other writer is mid-checkpoint.
    """
    for path in root.iterdir():
        if _PENDING_CHECKPOINT.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


def write_checkpoint(run_dir, trainer, last_batch, metadata):
    import torch
    root = Path(run_dir) / 'checkpoints'
    root.mkdir(exist_ok=True)
    sync_directory(root.parent)
    _sweep_pending(root)
    name = f"{metadata['attempt_id']}-step-{trainer.step:08d}-{uuid.uuid4().hex[:12]}"
    temporary, target = root / ('.pending-' + name), root / name
    temporary.mkdir()
    try:
        state = trainer_state(trainer, last_batch)
        with (temporary / 'state.pt').open('wb') as output:
            writer = _HashingWriter(output)
            torch.save(state, writer)
            output.flush()
            os.fsync(output.fileno())
        digest = writer.digest.hexdigest()
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
    import torch
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


from .model_file import DEFAULT_CHECKPOINT_POLICY, normalize_checkpoint_policy


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _durable_checkpoints(ckpt_root):
    """Completed checkpoint directories with their step and owning attempt, oldest first."""
    checkpoints = []
    if not Path(ckpt_root).is_dir():
        return checkpoints
    for d in Path(ckpt_root).iterdir():
        if not d.is_dir() or d.name.startswith('.'):
            continue
        info = _read_json(d / 'manifest.json') or {}
        step = info.get('step')
        if type(step) is not int:
            m = re.search(r'-step-(\d+)-', d.name)
            step = int(m.group(1)) if m else None
        if step is None:
            continue
        attempt = info.get('attempt_id')
        checkpoints.append({'name': d.name, 'path': d, 'step': step,
                            'attempt_id': attempt if isinstance(attempt, str) else None})
    checkpoints.sort(key=lambda c: (c['step'], c['name']))
    return checkpoints


def _run_policy(run_dir):
    manifest = _read_json(Path(run_dir) / 'manifest.json') or {}
    config = manifest.get('config')
    return config.get('checkpoints') if isinstance(config, dict) else None


def _rank_best(run_dir, evaluations, policy=None):
    from .summary import rank_evaluations
    return rank_evaluations(evaluations, _run_policy(run_dir) if policy is None else policy)


def _checkpoints_for_records(checkpoints, records):
    """Checkpoints holding each evaluated snapshot: same step and, when both are known, same attempt."""
    matches = {}
    for record in records:
        attempt = record.get('attempt_id')
        found = [c for c in checkpoints if c['step'] == record['step']
                 and (attempt is None or c['attempt_id'] is None or c['attempt_id'] == attempt)]
        matches[id(record)] = sorted(found, key=lambda c: c['name'])
    return matches


def _live_branch(run_dir, checkpoints, tip):
    """Names of checkpoints on the lineage that ends at ``tip``, or None when unknown.

    A resumed attempt continues from its ``recovery_parent``; checkpoints of an
    ancestor attempt after the step it was resumed from belong to an abandoned
    branch. Each attempt manifest records its own parent; a missing link makes
    the lineage unknown rather than cutting the branch short.
    """
    if tip is None or tip['attempt_id'] is None:
        return None
    parents, known = {}, set()
    manifests = [Path(run_dir) / 'manifest.json']
    attempts = Path(run_dir) / 'attempts'
    if attempts.is_dir():
        manifests.extend(path / 'manifest.json' for path in attempts.iterdir() if path.is_dir())
    for path in manifests:
        info = _read_json(path) or {}
        if isinstance(info.get('attempt_id'), str):
            known.add(info['attempt_id'])
        parent = info.get('recovery_parent')
        if (isinstance(info.get('attempt_id'), str) and isinstance(parent, dict)
                and isinstance(parent.get('attempt_id'), str) and type(parent.get('step')) is int):
            parents[info['attempt_id']] = (parent['attempt_id'], parent['step'])
    branch, seen = set(), set()
    attempt, limit = tip['attempt_id'], tip['step']
    while attempt is not None and attempt not in seen:
        if attempt not in known:
            return None
        seen.add(attempt)
        branch.update(c['name'] for c in checkpoints
                      if c['attempt_id'] in (attempt, None) and c['step'] <= limit)
        attempt, limit = parents.get(attempt, (None, None))
    return branch


CHECKPOINT_DIR_NAMES = ("checkpoints", "distributed-checkpoints")


def _checkpoint_roots(run_dir: Path, *, existing_only: bool = True) -> list[Path]:
    run_dir = Path(run_dir).resolve()
    roots = [(run_dir / name).resolve() for name in CHECKPOINT_DIR_NAMES]
    if existing_only:
        return [r for r in roots if r.is_dir()]
    return roots


def _checkpoint_sort_key(ckpt_dir: Path, run_manifest: dict | None = None) -> tuple:
    info = _read_json(ckpt_dir / "manifest.json") or {}
    step = info.get("step")
    if type(step) is not int:
        m = re.search(r"-step-(\d+)-", ckpt_dir.name)
        step = int(m.group(1)) if m else -1
    eb = info.get("event_boundary") if isinstance(info.get("event_boundary"), dict) else {}
    seq = eb.get("sequence", -1) if type(eb.get("sequence")) is int else -1
    manifest_ckpt = (run_manifest or {}).get("checkpoint_path")
    is_current = False
    if isinstance(manifest_ckpt, str) and manifest_ckpt:
        try:
            is_current = ckpt_dir.resolve() == Path(manifest_ckpt).resolve()
        except OSError:
            is_current = False
    try:
        mtime_ns = (ckpt_dir / "manifest.json").stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    return (step, seq, is_current, mtime_ns, ckpt_dir.name)


def resolve_resume_checkpoint_dir(run_dir: Path | str, checkpoint=None, *, run_manifest: dict | None = None) -> tuple[Path, dict | None]:
    """Resolve latest or explicit checkpoint across checkpoints/ and distributed-checkpoints/."""
    run_dir = Path(run_dir).resolve()
    all_roots = _checkpoint_roots(run_dir, existing_only=False)
    if run_manifest is None:
        run_manifest = _read_json(run_dir / "manifest.json") or {}

    is_latest = checkpoint is None or (isinstance(checkpoint, str) and checkpoint.strip().lower() == "latest")
    if is_latest:
        candidates = []
        for root in all_roots:
            pointer_path = root / "latest.json"
            if not pointer_path.exists():
                continue
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("Invalid training checkpoint pointer") from exc
            if not isinstance(pointer, dict) or type(pointer.get("schema_version")) is not int or pointer["schema_version"] != SCHEMA:
                raise ValueError("Unsupported training checkpoint pointer schema")
            if root.name == "distributed-checkpoints" and pointer.get("kind") != "hypergan-distributed-training-checkpoint":
                raise ValueError("Invalid distributed checkpoint pointer kind")
            name = pointer.get("checkpoint")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Invalid training checkpoint pointer path")
            cand = root / name
            if cand.is_symlink():
                raise ValueError("Checkpoint must be an ordinary completed directory inside this run")
            cand_resolved = cand.resolve()
            if cand_resolved.parent != root or cand_resolved.name.startswith(".") or not cand_resolved.is_dir():
                raise ValueError("Checkpoint must be a completed checkpoint directory inside this run")
            info = _read_json(cand_resolved / "manifest.json")
            if not isinstance(info, dict):
                raise ValueError("Invalid training checkpoint metadata")
            step = info.get("step")
            if type(step) is not int:
                raise ValueError("Checkpoint step is outside the original schedule")
            if type(pointer.get("step")) is not int:
                raise ValueError("Checkpoint pointer step differs from generation")
            candidates.append((_checkpoint_sort_key(cand_resolved, run_manifest), cand_resolved, pointer))
        if not candidates:
            raise ValueError("No full training checkpoint exists; an inference model is not resumable")
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, winner, winner_pointer = candidates[0]
        return winner, winner_pointer

    # Explicit step int, digit string, or 'best'
    if (
        (isinstance(checkpoint, int) and not isinstance(checkpoint, bool))
        or (isinstance(checkpoint, str) and checkpoint.strip() == "best")
        or (
            isinstance(checkpoint, str)
            and checkpoint.strip().isdigit()
            and not (run_dir / checkpoint).exists()
            and not any((r / checkpoint).exists() for r in all_roots)
        )
    ):
        resolved = resolve_checkpoint(run_dir, checkpoint)
        return resolved, None

    target = Path(checkpoint)
    if not target.is_absolute():
        choices = (target, run_dir / target, *(r / target for r in all_roots))
        target = next((choice for choice in choices if choice.exists() or choice.is_symlink()), all_roots[0] / target)
    if target.is_symlink():
        raise ValueError("Checkpoint must be an ordinary completed directory inside this run")
    target = target.resolve()
    if target.parent not in all_roots or target.name.startswith(".") or not target.is_dir():
        raise ValueError("Checkpoint must be a completed checkpoint directory inside this run")
    return target, None


def resolve_checkpoint(run_dir: Path | str, target: str | int | None = "latest") -> Path:
    """Resolve a checkpoint target ('latest', 'best', integer step, or path) to a checkpoint directory."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory not found: {run_dir}")
    ckpt_roots = _checkpoint_roots(run_dir, existing_only=True)
    if not ckpt_roots:
        raise ValueError(f"No checkpoints directory found in {run_dir}")
    run_manifest = _read_json(run_dir / "manifest.json") or {}

    target_str = str(target).strip() if target is not None else "latest"

    # 1. Target: "latest" (or None)
    if target is None or target_str in ("latest", "None", ""):
        pointer_candidates = []
        for ckpt_root in ckpt_roots:
            latest_file = ckpt_root / "latest.json"
            if latest_file.is_file():
                try:
                    latest = json.loads(latest_file.read_text(encoding="utf-8"))
                    if isinstance(latest, dict) and latest.get("checkpoint"):
                        candidate = ckpt_root / latest["checkpoint"]
                        if candidate.is_dir():
                            resolved_cand = candidate.resolve()
                            pointer_candidates.append((_checkpoint_sort_key(resolved_cand, run_manifest), resolved_cand))
                except Exception:
                    pass
        if pointer_candidates:
            pointer_candidates.sort(key=lambda item: item[0], reverse=True)
            return pointer_candidates[0][1]

        # Fallback to max step on disk across all roots
        candidates = []
        for ckpt_root in ckpt_roots:
            for p in ckpt_root.iterdir():
                if p.is_dir() and not p.name.startswith("."):
                    match = re.search(r"-step-(\d+)-", p.name)
                    step_val = None
                    if match:
                        step_val = int(match.group(1))
                    else:
                        mf = p / "manifest.json"
                        if mf.is_file():
                            try:
                                m_info = json.loads(mf.read_text(encoding="utf-8"))
                                step_val = m_info.get("step")
                            except Exception:
                                pass
                    if step_val is not None:
                        resolved_p = p.resolve()
                        candidates.append((_checkpoint_sort_key(resolved_p, run_manifest), resolved_p))
        if not candidates:
            raise ValueError(f"No training checkpoints found in {ckpt_roots[0]}")
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    # 2. Target: "best": the best-ranked primary-metric evaluation that has a checkpoint
    if target_str == "best":
        from .summary import collect_evaluations
        evaluations = collect_evaluations(run_dir)
        if not evaluations:
            raise ValueError("No evaluation records available to determine best checkpoint; pass explicit step or latest")
        metric, _, ranked = _rank_best(run_dir, evaluations)
        if metric is None:
            raise ValueError("No primary metric defines the best checkpoint: set checkpoints.keep_best.metric "
                             "to an evaluated metric, or pass an explicit step or latest")
        all_durable = []
        for ckpt_root in ckpt_roots:
            all_durable.extend(_durable_checkpoints(ckpt_root))
        all_durable.sort(key=lambda c: (c["step"], c["name"]))
        by_record = _checkpoints_for_records(all_durable, ranked)
        for record in ranked:
            if by_record.get(id(record)):
                return by_record[id(record)][-1]['path'].resolve()
        raise ValueError(f"No checkpoint exists at any evaluated step of {metric} in {ckpt_roots[0]}")

    # 3. Target: STEP (integer or numeric digits)
    if (isinstance(target, int) and not isinstance(target, bool)) or target_str.isdigit():
        step = int(target)
        pattern = f"-step-{step:08d}-"
        matches = [
            p for ckpt_root in ckpt_roots for p in ckpt_root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and pattern in p.name
        ]
        if not matches:
            fallback_pattern = f"-step-{step}-"
            matches = [
                p for ckpt_root in ckpt_roots for p in ckpt_root.iterdir()
                if p.is_dir() and not p.name.startswith(".") and fallback_pattern in p.name
            ]
        if not matches:
            for ckpt_root in ckpt_roots:
                for p in ckpt_root.iterdir():
                    if p.is_dir() and not p.name.startswith("."):
                        mf = p / "manifest.json"
                        if mf.is_file():
                            try:
                                m_info = json.loads(mf.read_text(encoding="utf-8"))
                                if m_info.get("step") == step:
                                    matches.append(p)
                            except Exception:
                                pass
        if not matches:
            raise ValueError(f"Checkpoint for step {step} not found in {ckpt_roots[0]}")
        matches.sort(key=lambda p: _checkpoint_sort_key(p.resolve(), run_manifest), reverse=True)
        return matches[0].resolve()

    # 4. Target: PATH
    cand_path = Path(target_str)
    choices = [cand_path, run_dir / cand_path, *(ckpt_root / cand_path for ckpt_root in ckpt_roots)]
    for c in choices:
        if c.is_file() and c.name in ("state.pt", "model.pt", "rank-00000.pt"):
            return c.parent.resolve()
        if c.is_dir() and (
            (c / "state.pt").is_file()
            or (c / "rank-00000.pt").is_file()
            or (c / "manifest.json").is_file()
            or (c / "model.pt").is_file()
        ):
            return c.resolve()

    raise ValueError(f"Checkpoint path not found: {target}")


def extract_ema_bundle(checkpoint_dir: Path | str, manifest: dict | None = None, include_example_inputs: bool = False) -> dict:
    """Extract an in-memory EMA inference bundle conforming to kind='ema-inference' from a checkpoint."""
    import torch
    from .config import config_values, sampling_bindings, evaluation_bindings
    path = Path(checkpoint_dir).resolve()
    if path.is_file() and path.name == "model.pt":
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(bundle, dict) and bundle.get("kind") == "ema-inference":
            if not include_example_inputs and "example_inputs" in bundle:
                bundle = dict(bundle)
                bundle["example_inputs"] = {
                    k: ({"shape": list(v.shape), "dtype": str(v.dtype)} if hasattr(v, "shape") else v)
                    for k, v in bundle["example_inputs"].items()
                }
            return bundle
        state_file = path
        ckpt_dir = path.parent
    elif path.is_file() and path.name in ("state.pt", "rank-00000.pt"):
        state_file = path
        ckpt_dir = path.parent
    elif path.is_dir():
        ckpt_dir = path
        state_file = ckpt_dir / "state.pt"
        if not state_file.is_file() and (ckpt_dir / "rank-00000.pt").is_file():
            state_file = ckpt_dir / "rank-00000.pt"
        if not state_file.is_file() and (ckpt_dir / "model.pt").is_file():
            bundle = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
            if isinstance(bundle, dict) and bundle.get("kind") == "ema-inference":
                if not include_example_inputs and "example_inputs" in bundle:
                    bundle = dict(bundle)
                    bundle["example_inputs"] = {
                        k: ({"shape": list(v.shape), "dtype": str(v.dtype)} if hasattr(v, "shape") else v)
                        for k, v in bundle["example_inputs"].items()
                    }
                return bundle
    else:
        raise ValueError(f"Checkpoint path not found: {checkpoint_dir}")

    if not state_file.is_file():
        raise ValueError(f"Checkpoint state file missing: {state_file}")

    if manifest is None:
        manifest_file = ckpt_dir / "manifest.json"
        if not manifest_file.is_file():
            if (ckpt_dir.parent / "manifest.json").is_file():
                manifest_file = ckpt_dir.parent / "manifest.json"
            else:
                raise ValueError(f"Checkpoint manifest missing: {ckpt_dir / 'manifest.json'}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    state = torch.load(state_file, map_location="cpu", weights_only=True)

    if not isinstance(state, dict):
        raise ValueError("Checkpoint state missing EMA weights; cannot construct EMA inference bundle")
    if "ema_graph" not in state or "ema_prior" not in state:
        raise ValueError("Checkpoint state missing EMA weights; cannot construct EMA inference bundle")
    if state["ema_graph"] is None or state["ema_prior"] is None:
        raise ValueError("Checkpoint state missing EMA weights; cannot construct EMA inference bundle")

    config = manifest.get("config")
    if not isinstance(config, dict) and isinstance(manifest.get("identity"), dict):
        run_manifest = _read_json(ckpt_dir.parent.parent / "manifest.json") or {}
        if isinstance(run_manifest.get("config"), dict) and (
            not manifest.get("run_id") or run_manifest.get("run_id") == manifest.get("run_id")
        ):
            config = run_manifest["config"]
        elif isinstance(manifest["identity"].get("config"), dict):
            from .config import resolve_config
            config = resolve_config(manifest["identity"]["config"])
    if not isinstance(config, dict) or "components" not in config:
        raise ValueError("Checkpoint manifest missing valid configuration components")

    needed_components = set()
    called_components = set()

    def include(name):
        if name in called_components or name not in config["components"]:
            return
        called_components.add(name)
        needed_components.add(name)
        comp_spec = config["components"][name]
        if "reuse" in comp_spec:
            needed_components.add(comp_spec["reuse"])
        for input_path in comp_spec.get("inputs", {}).values():
            if isinstance(input_path, str) and input_path.startswith("components."):
                include(input_path.split(".")[1])

    include("generator")
    for binding in sampling_bindings(config.get("sampling", {})):
        if binding.startswith("components."):
            include(binding.split(".")[1])
    sampling_components = set(called_components)
    if isinstance(config.get("metrics"), dict) and isinstance(config["metrics"].get("custom"), dict):
        for binding in evaluation_bindings(config):
            if binding.startswith("components."):
                include(binding.split(".")[1])

    specs = {name: spec for name, spec in config["components"].items() if name in needed_components}
    needed_inputs = {
        path.split(".")[1]
        for name in sampling_components
        for path in specs[name].get("inputs", {}).values()
        if isinstance(path, str) and path.startswith("batch.")
    }

    raw_ema_graph = state["ema_graph"]
    raw_buffers = state.get("buffers", {}).get("ema_graph", {}) if isinstance(state.get("buffers"), dict) else {}

    model_states = {}
    model_buffers = {}
    for name, spec in specs.items():
        if "reuse" not in spec:
            prefix = f"models.{name}."
            matching_weights = {
                k[len(prefix):]: v for k, v in raw_ema_graph.items() if k.startswith(prefix)
            }
            if not matching_weights and name in raw_ema_graph and isinstance(raw_ema_graph[name], dict):
                matching_weights = raw_ema_graph[name]
            model_states[name] = matching_weights

            matching_buffers = {
                k[len(prefix):]: v for k, v in raw_buffers.items() if k.startswith(prefix)
            }
            if not matching_buffers and name in raw_buffers and isinstance(raw_buffers[name], dict):
                matching_buffers = raw_buffers[name]
            model_buffers[name] = matching_buffers

    prior_state = state["ema_prior"]
    prior_buffers = state.get("buffers", {}).get("ema_prior", {}) if isinstance(state.get("buffers"), dict) else {}

    batch = state.get("last_batch") or {}
    raw_inputs = {k: v for k, v in batch.items() if k in needed_inputs} if isinstance(batch, dict) else {}
    if not include_example_inputs:
        example_inputs = {}
        for k, v in raw_inputs.items():
            if hasattr(v, "shape"):
                example_inputs[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                example_inputs[k] = v
    else:
        example_inputs = raw_inputs

    identity = {
        "run_id": manifest.get("run_id", ""),
        "attempt_id": manifest.get("attempt_id", ""),
        "source_step": state.get("step", manifest.get("step", 0)),
    }
    if "identity" in manifest and isinstance(manifest["identity"], dict):
        identity.update(manifest["identity"])

    bundle = {
        "schema_version": 1,
        "kind": "ema-inference",
        "resume_supported": False,
        "step": state.get("step", manifest.get("step", 0)),
        "config": config_values(config),
        "components": specs,
        "model_states": model_states,
        "prior": prior_state,
        "example_inputs": example_inputs,
        "model_buffers": model_buffers,
        "prior_buffers": prior_buffers,
        "identity": identity,
    }
    return bundle


def export_checkpoint_bundle(checkpoint_dir: Path | str, output_dir: Path | str, include_example_inputs: bool = False) -> tuple[Path, Path]:
    """Export an EMA inference bundle (model.pt + model.sha256) to output directory or file path."""
    import torch
    ckpt_dir = Path(checkpoint_dir).resolve()
    bundle = extract_ema_bundle(ckpt_dir, include_example_inputs=include_example_inputs)

    dest = Path(output_dir).resolve()
    if dest.is_dir() or (not dest.exists() and dest.suffix == ""):
        dest.mkdir(parents=True, exist_ok=True)
        dest = dest / "model.pt"
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)

    temp_pt = dest.with_suffix(".pt.tmp") if dest.suffix == ".pt" else dest.with_name(dest.name + ".tmp")
    with temp_pt.open("wb") as stream:
        torch.save(bundle, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp_pt.replace(dest)

    digest = file_sha256(dest)
    checksum_path = dest.with_name("model.sha256") if dest.name == "model.pt" else dest.with_suffix(".sha256")
    temp_sha = checksum_path.with_suffix(".sha256.tmp") if checksum_path.suffix == ".sha256" else checksum_path.with_name(checksum_path.name + ".tmp")
    with temp_sha.open("w", encoding="utf-8") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp_sha.replace(checksum_path)
    sync_directory(dest.parent)
    return dest, checksum_path


def _prune_report(run_dir, dry_run, retained=(), deleted=()):
    retained, deleted = list(retained), list(deleted)
    reclaimed = sum(item.get('size', 0) for item in deleted)
    return {
        'run_dir': str(run_dir),
        'dry_run': dry_run,
        'retained_count': len(retained),
        'deleted_count': len(deleted),
        'retained': retained,
        'deleted': deleted,
        'reclaimed_bytes': reclaimed,
        # aliases:
        'pruned': deleted,
        'pruned_count': len(deleted),
        'bytes_reclaimed': reclaimed,
    }


@contextmanager
def _prune_lock(ckpt_root: Path):
    lock_path = ckpt_root / '.prune.lock'
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        lock_file(fd, blocking=True)
        try:
            yield
        finally:
            unlock_file(fd)
    finally:
        os.close(fd)


def prune_checkpoints(
    run_dir: Path | str,
    policy: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Evaluate and apply checkpoint retention policy to a run directory adhering to Invariants S1-S5.

    ``keep_last`` counts checkpoints on the live branch, the lineage that ends
    at ``latest.json``. After a resume from an older checkpoint, each abandoned
    attempt keeps its newest checkpoint, so the old branch stays resumable;
    its other checkpoints keep only milestone, best, step-zero and
    active-evaluation protection.
    """
    try:
        from .registry import resolve_destructive_target
        run_dir = resolve_destructive_target(run_dir)
    except (ValueError, FileNotFoundError):
        run_dir = Path(run_dir).resolve()

    if not run_dir.is_dir():
        return _prune_report(run_dir, dry_run)

    ckpt_roots = _checkpoint_roots(run_dir, existing_only=True)
    if not ckpt_roots:
        return _prune_report(run_dir, dry_run)

    with _prune_lock(ckpt_roots[0]):
        # 1. Resolve Policy
        if policy is None:
            policy = _run_policy(run_dir)
        policy = normalize_checkpoint_policy(policy)

        # 2. Discover durable checkpoints across checkpoints/ and distributed-checkpoints/
        checkpoints = []
        for ckpt_root in ckpt_roots:
            checkpoints.extend(_durable_checkpoints(ckpt_root))
        checkpoints.sort(key=lambda c: (c['step'], c['name']))
        for c in checkpoints:
            size = 0
            try:
                for item in c['path'].rglob('*'):
                    if item.is_file():
                        size += item.stat().st_size
            except OSError:
                pass
            c['size'] = size

        if len(checkpoints) <= 1:
            retained = [{'step': c['step'], 'name': c['name'], 'reasons': ['single_checkpoint']} for c in checkpoints]
            return _prune_report(run_dir, dry_run, retained)

        # 3. Retain map: name -> set of reasons
        retained_reasons = defaultdict(set)

        # S1: Never delete latest. Compare pointers across roots so the newest tip wins.
        m_data = _read_json(run_dir / 'manifest.json') or {}
        tip_candidates = []
        root_latest_names = {}
        for ckpt_root in ckpt_roots:
            latest_name = (_read_json(ckpt_root / 'latest.json') or {}).get('checkpoint')
            if latest_name:
                root_latest_names[ckpt_root] = latest_name
                cand = next((c for c in checkpoints if c['path'].parent == ckpt_root and c['name'] == latest_name), None)
                if cand is not None:
                    tip_candidates.append((_checkpoint_sort_key(cand['path'], m_data), cand))
        tip_candidates.sort(key=lambda item: item[0], reverse=True)
        tip = tip_candidates[0][1] if tip_candidates else None
        branch = _live_branch(run_dir, checkpoints, tip)
        max_step = max(c['step'] for c in checkpoints)
        for c in checkpoints:
            if c is tip or (tip is None and c['step'] == max_step):
                retained_reasons[c['name']].add('latest')
        live = [c for c in checkpoints if branch is None or c['name'] in branch]
        live_names = {c['name'] for c in live}

        # S2: Pin active interval evaluations
        pinned_steps = set()
        schedule = m_data.get('evaluation_schedule')
        for item in (schedule.values() if isinstance(schedule, dict) else []):
            if isinstance(item, dict) and item.get('status') == 'running' and type(item.get('source_step')) is int:
                pinned_steps.add(item['source_step'])

        eval_root = run_dir / 'metrics' / 'evaluations'
        if eval_root.is_dir():
            for r_file in eval_root.glob('*/receipt.json'):
                receipt = _read_json(r_file) or {}
                if receipt.get('status') == 'running':
                    s = receipt.get('source_step') if receipt.get('source_step') is not None else receipt.get('step')
                    if type(s) is int:
                        pinned_steps.add(s)

        for c in checkpoints:
            if c['step'] in pinned_steps:
                retained_reasons[c['name']].add('active_evaluation')

        # S3: Retain step 0 (initial state)
        for c in checkpoints:
            if c['step'] == 0:
                retained_reasons[c['name']].add('step_zero')

        # S4: Keep milestones (keep_every)
        keep_every = policy.get('keep_every')
        if keep_every and keep_every > 0:
            for c in checkpoints:
                if c['step'] % keep_every == 0:
                    retained_reasons[c['name']].add(f'milestone(step={c["step"]})')

        # S5: Keep the best checkpoints of the run's one primary metric, ranked
        # exactly as summary "best" and resolve_checkpoint('best') rank them.
        keep_best = policy.get('keep_best')
        if isinstance(keep_best, dict) and keep_best.get('count', 0) > 0:
            from .summary import collect_evaluations
            metric, _, ranked = _rank_best(run_dir, collect_evaluations(run_dir), policy)
            matches = _checkpoints_for_records(checkpoints, ranked)
            remaining = keep_best['count']
            for record in ranked:
                names = [c['name'] for c in matches[id(record)] if f'best({metric})' not in retained_reasons[c['name']]]
                if not names:
                    continue
                for name in names:
                    retained_reasons[name].add(f'best({metric})')
                remaining -= 1
                if remaining == 0:
                    break

        # Keep last N checkpoints of the live branch
        keep_last = policy.get('keep_last', 3)
        if keep_last > 0:
            last_steps = set(sorted({c['step'] for c in live})[-keep_last:])
            recovery_ckpt = (m_data.get('recovery_parent') or {}).get('checkpoint_id')
            for c in checkpoints:
                if (c['name'] in live_names and c['step'] in last_steps) or (recovery_ckpt and c['name'] == recovery_ckpt):
                    retained_reasons[c['name']].add('keep_last')

        # The newest checkpoint of each attempt off the live branch stays as that
        # abandoned branch's tip; it no longer counts toward keep_last.
        abandoned = {}
        for c in checkpoints:
            if c['name'] not in live_names:
                abandoned[c['attempt_id']] = c
        for c in abandoned.values():
            retained_reasons[c['name']].add('abandoned_tip')

        # S3: Require a newer durable checkpoint before deleting older. Off the
        # live branch, the retained live tip is that newer durable checkpoint.
        deleted_candidates = []
        for c in checkpoints:
            if c['name'] in retained_reasons:
                continue
            has_newer = c['name'] not in live_names or any(other['step'] > c['step'] for other in live)
            if not has_newer:
                retained_reasons[c['name']].add('s3_no_newer_durable')
            else:
                deleted_candidates.append(c)

        # Execute deletion if not dry_run
        deleted = []
        if not dry_run:
            deleted_names_by_root = defaultdict(set)
            for c in deleted_candidates:
                target_dir = c['path']
                try:
                    (target_dir / 'state.pt').unlink(missing_ok=True)
                    (target_dir / 'manifest.json').unlink(missing_ok=True)
                    shutil.rmtree(target_dir)
                    deleted.append({'step': c['step'], 'name': c['name'], 'size': c['size']})
                    deleted_names_by_root[target_dir.parent].add(c['name'])
                except OSError:
                    pass
            if deleted:
                for ckpt_root, removed_names in deleted_names_by_root.items():
                    if root_latest_names.get(ckpt_root) in removed_names:
                        remaining_in_root = [
                            c for c in checkpoints
                            if c['path'].parent == ckpt_root and c['name'] not in removed_names
                        ]
                        pointer_file = ckpt_root / 'latest.json'
                        if remaining_in_root:
                            newest = max(remaining_in_root, key=lambda c: _checkpoint_sort_key(c['path'], m_data))
                            old_ptr = _read_json(pointer_file) or {'schema_version': SCHEMA}
                            new_ptr = dict(old_ptr, schema_version=SCHEMA, checkpoint=newest['name'], step=newest['step'])
                            atomic_json(pointer_file, new_ptr)
                        else:
                            pointer_file.unlink(missing_ok=True)
                    sync_directory(ckpt_root)
        else:
            deleted = [
                {'step': c['step'], 'name': c['name'], 'size': c['size']}
                for c in deleted_candidates
            ]

        retained = [
            {'step': c['step'], 'name': c['name'], 'reasons': sorted(list(retained_reasons[c['name']]))}
            for c in checkpoints if c['name'] in retained_reasons
        ]
        return _prune_report(run_dir, dry_run, retained, deleted)
