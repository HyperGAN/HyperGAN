"""Read-only online checkpoint reconstruction for diagnostics, not training resume."""
import copy

import torch

from .checkpoints import data_contract, read_checkpoint, restore_rng
from .checkpoint_compatibility import validate_implementation
from .config import config_values, resolve_config
from .execution import prepare_resume
from .training import ReferenceTrainer, _implementation, runtime_info


def load_signal_checkpoint(run_dir, checkpoint=None, *, device=None, batch_size=None):
    """Pin complete G/D state and load it on the requested evaluation device.

    Training's runtime/topology continuation contract is deliberately untouched.
    This separate loader never restores optimizer state, writes a run, or invokes
    an update. Compatible saved RNG streams are preserved; any backend-dependent
    reset is explicit in the returned diagnostic protocol.
    """
    prepared = prepare_resume(run_dir, checkpoint=checkpoint)
    if prepared.profile is not None and prepared.profile['execution']['name'] != 'cpu-single':
        raise ValueError('Checkpoint signal diagnosis currently supports native single-process runs')
    target, info, state = read_checkpoint(prepared.run_dir, prepared.checkpoint)
    source_config = prepared.config
    values = config_values(source_config)
    if device is not None:
        values['training']['device'] = device
    if batch_size is not None:
        values['training']['batch_size'] = batch_size
    config = resolve_config(values)
    required = {'graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs',
                'step', 'streams', 'rng', 'data', 'modes', 'buffers', 'trainable', 'last_batch'}
    if set(state) != required or not isinstance(state['optimizers'], list) or len(state['optimizers']) != 2:
        raise ValueError('Signal diagnosis requires a complete online training checkpoint')
    trainer = ReferenceTrainer(config)
    validate_implementation(info['implementation'], _implementation(trainer))
    contract = data_contract(trainer.data, config['data'])
    if not contract['supported'] or contract != info['data_contract']:
        raise ValueError('Diagnostic data identity or recovery protocol differs from checkpoint')
    for name in ('graph', 'prior'):
        module = getattr(trainer, name)
        current = module.state_dict()
        if not isinstance(state[name], dict) or set(current) != set(state[name]):
            raise ValueError('Checkpoint online model state inventory differs from diagnostic model')
        for key, value in current.items():
            saved = state[name][key]
            if isinstance(value, torch.Tensor) and (
                    not isinstance(saved, torch.Tensor) or saved.shape != value.shape or saved.dtype != value.dtype):
                raise ValueError('Checkpoint online model tensor shape or dtype differs from diagnostic model')
        buffers = dict(module.named_buffers())
        parameters = dict(module.named_parameters())
        modules = dict(module.named_modules())
        if (set(state['buffers'][name]) != set(buffers) or set(state['trainable'][name]) != set(parameters)
                or set(state['modes'][name]) != set(modules)):
            raise ValueError('Checkpoint online buffer, parameter or module inventory differs')
        if any(type(value) is not bool for value in (*state['trainable'][name].values(), *state['modes'][name].values())):
            raise ValueError('Checkpoint module modes and trainability flags must be booleans')
        for key, value in buffers.items():
            saved = state['buffers'][name][key]
            if not isinstance(saved, torch.Tensor) or saved.shape != value.shape or saved.dtype != value.dtype:
                raise ValueError('Checkpoint online buffer shape or dtype differs')
        module.load_state_dict(state[name], strict=True)
        with torch.no_grad():
            for key, value in buffers.items():
                value.copy_(state['buffers'][name][key])
        for key, value in parameters.items():
            value.requires_grad_(state['trainable'][name][key])
        for key, value in modules.items():
            value.training = state['modes'][name][key]
    trainer.step = state['step']
    if contract['stateful']:
        if state['data'] is None:
            raise ValueError('Checkpoint is missing declared stateful data state')
        trainer.data.load_state_dict(copy.deepcopy(state['data']))
    elif state['data'] is not None:
        raise ValueError('Stateless diagnostic data has unexpected checkpoint state')
    if not isinstance(state['streams'], dict) or set(state['streams']) != set(trainer.streams):
        raise ValueError('Checkpoint named RNG stream inventory differs from diagnostic model')
    streams = {}
    for name, stream in trainer.streams.items():
        saved = state['streams'][name]
        if not isinstance(saved, torch.Tensor) or saved.dtype != torch.uint8 or saved.ndim != 1:
            raise ValueError('Invalid checkpoint named RNG state')
        if saved.shape == stream.get_state().shape:
            stream.set_state(saved)
            streams[name] = 'restored'
        else:
            streams[name] = 'reset-from-config-backend-incompatible'
    # CPU/Python/NumPy are device independent. Never set other CUDA devices or
    # require the original GPU inventory merely to evaluate immutable weights.
    restore_rng({key: value for key, value in state['rng'].items() if key not in ('cuda', 'cuda_device')})
    cuda_rng = 'not-applicable'
    if trainer.device.type == 'cuda':
        saved_device = torch.device(info['runtime']['device'])
        index = saved_device.index if saved_device.index is not None else state['rng'].get('cuda_device')
        saved_cuda = state['rng'].get('cuda')
        if (saved_device.type == 'cuda' and type(index) is int and isinstance(saved_cuda, list)
                and 0 <= index < len(saved_cuda) and isinstance(saved_cuda[index], torch.Tensor)
                and saved_cuda[index].shape == torch.cuda.get_rng_state(trainer.device).shape):
            torch.cuda.set_rng_state(saved_cuda[index], trainer.device)
            cuda_rng = 'restored-source-training-device-to-evaluation-device'
        else:
            with torch.cuda.device(trainer.device):
                torch.cuda.manual_seed(config['training']['seed'])
            cuda_rng = 'reset-from-config-backend-incompatible'
    protocol = {'weights': 'online-generator-and-discriminator', 'optimizer_state_loaded': False,
                'optimizer_steps': 0, 'checkpoint_base_lrs': copy.deepcopy(state['base_lrs']),
                'probe_draw': 'next saved checkpoint draw where RNG backends are compatible; configuration initialization otherwise',
                'data_state': 'restored' if contract['stateful'] else 'stateless',
                'named_rng_streams': streams, 'global_cpu_rng': 'restored', 'global_cuda_rng': cuda_rng,
                'source_runtime': info['runtime'], 'evaluation_runtime': runtime_info(trainer.device),
                'interpretation': 'Read-only diagnostic reconstruction; no claim of bitwise training continuation across runtimes or devices.'}
    return trainer, source_config, target, info, protocol
