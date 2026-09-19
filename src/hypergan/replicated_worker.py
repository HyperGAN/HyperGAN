"""Numerical side of the internal replicated execution adapter.

Only the supervised rank imports this module. No handler publishes a canonical
training checkpoint; the parent owns that authority and the run lock.

Inference has a command deadline and post-write file size checks, not a sandbox
or an allocation bound for custom Python. Rank zero retains its Gloo group;
collective-dependent inference fails the job rather than involving idle peers.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

from .checkpoints import capture_rng, restore_rng
from .config import resolve_config
from .distributed_checkpoints import (distributed_checkpoint_identity,
    prepare_distributed_checkpoint, restore_distributed_checkpoint)
from .distributed_training import ReplicatedCPUTrainer
from .training import _recovery_contract, runtime_info, source_info
from .replicated_execution import MAX_SAMPLE_COUNT

MAX_INFORMATION_BYTES = 40 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_SAMPLE_BYTES = 2 * 1024 * 1024


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _ready(state):
    trainer = state['trainer']
    if not trainer.checkpoint_ready or trainer._poisoned:
        raise RuntimeError('Replicated execution is not at a complete update boundary')
    if torch.get_num_threads() != 1 or torch.get_default_dtype() != torch.float32:
        raise ValueError('Replicated execution requires one CPU thread and float32 default dtype')
    return {'step': trainer.step, 'ready': True, 'inference_available': state['batch'] is not None}


def create_worker(rank, world_size, config, execution, context):
    trainer = ReplicatedCPUTrainer(resolve_config(config), world_size=world_size,
                                    accumulation_steps=execution['accumulation_steps'])
    if any(trainer.strategy_info.get(key) != value for key, value in execution.items()):
        raise ValueError('Actual replicated strategy differs from resolved execution profile')
    return {'rank': rank, 'trainer': trainer, 'batch': None, 'context': context}


def _information(state):
    trainer = state['trainer']
    rng = capture_rng()
    streams = {name: stream.get_state() for name, stream in trainer.streams.items()}
    try:
        contract, reasons = _recovery_contract(trainer)
        identity = distributed_checkpoint_identity(trainer) if not reasons else None
        runtime = runtime_info()
        runtime.update(world_size=trainer.world_size, backend='gloo', threads=torch.get_num_threads(),
                       interop_threads=torch.get_num_interop_threads())
        info = {'data_identity': contract['identity'], 'recovery_reasons': reasons,
                'identity': identity, 'environment': {'runtime': runtime, 'source': source_info()}}
        if len(_encoded(info)) > MAX_INFORMATION_BYTES:
            raise ValueError(f'Replicated execution identity exceeds the {MAX_INFORMATION_BYTES}-byte command information limit')
        _ready(state)
        return info
    finally:
        restore_rng(rng)
        for name, value in streams.items():
            trainer.streams[name].set_state(value)


def _inference(state, payload):
    trainer = state['trainer']
    context = state['context']
    directory = Path(context['attempt_dir']) / 'inference'
    if payload['bundle_dir'] != str(directory) or payload['identity']['run_id'] != context['run_id'] or payload['identity']['attempt_id'] != context['attempt_id']:
        raise ValueError('Inference destination or identity differs from the current attempt')
    if state['batch'] is None:
        raise ValueError('Inference requires a completed training batch')
    if trainer.config['sampling']['count'] > MAX_SAMPLE_COUNT:
        raise ValueError(f'Replicated final inference sample count exceeds {MAX_SAMPLE_COUNT}; no samples were truncated')
    if state['rank']:
        return _ready(state)  # Peers wait outside collectives while rank zero renders.
    from .artifacts import save_bundle, sample
    rng, threads = capture_rng(), torch.get_num_threads()
    streams = {name: stream.get_state() for name, stream in trainer.streams.items()}
    try:
        # Module singleton dependencies are not mutable owned numerical state.
        memo = {id(module): module for module in list(sys.modules.values()) if isinstance(module, ModuleType)}
        snapshot = SimpleNamespace(config=copy.deepcopy(trainer.config), step=trainer.step,
            ema_graph=copy.deepcopy(trainer.ema_graph, memo), ema_prior=copy.deepcopy(trainer.ema_prior, memo),
            artifact_identity=copy.deepcopy(payload['identity']))
        batch = copy.deepcopy(state['batch'])
        save_bundle(directory, snapshot, batch)
        if (directory / 'model.pt').stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError(f'Replicated inference bundle exceeds {MAX_BUNDLE_BYTES} bytes')
        path = sample(directory, count=trainer.config['sampling']['count'], seed=trainer.config['sampling']['seed'])
        if path.stat().st_size > MAX_SAMPLE_BYTES:
            raise ValueError(f'Replicated inference sample exceeds {MAX_SAMPLE_BYTES} bytes')
        return {**_ready(state), 'bundle_path': str(directory / 'model.pt'), 'sample_path': str(path)}
    finally:
        restore_rng(rng)
        for name, value in streams.items():
            trainer.streams[name].set_state(value)
        torch.set_num_threads(threads)


def handle_command(state, operation, payload):
    trainer, context = state['trainer'], state['context']
    _ready(state)
    if operation == 'describe':
        info = trainer._phase('execution identity', lambda: _information(state))
        digest = hashlib.sha256(_encoded(info)).hexdigest()
        peers = trainer._exchange('execution identity digest', digest)
        if any(peer != digest for peer in peers):
            raise ValueError('Ranks disagree on execution runtime/source/data identity')
        return {**_ready(state), 'identity_sha256': digest, **({'information': info} if state['rank'] == 0 else {})}
    if operation == 'update':
        metrics, state['batch'] = trainer.update()
        return {**_ready(state), 'metrics': metrics}
    if operation == 'restore':
        path, info, state['batch'] = restore_distributed_checkpoint(context['run_dir'], trainer,
            {'run_id': context['run_id']}, checkpoint=payload['checkpoint'])
        return {**_ready(state), 'checkpoint_path': str(path), 'saved_step': info['step']}
    if operation == 'prepare':
        metadata = payload['metadata']
        if any(metadata[key] != context[key] for key in ('run_id', 'attempt_id')):
            raise ValueError('Checkpoint lineage differs from the immutable worker attempt')
        receipt = prepare_distributed_checkpoint(context['run_dir'], trainer, state['batch'], metadata,
            command_sequence=payload['command_sequence'], controller_id=payload['controller_id'])
        digest = hashlib.sha256(_encoded(receipt)).hexdigest()
        return {**_ready(state), 'receipt_sha256': digest, **({'receipt': receipt} if state['rank'] == 0 else {})}
    if operation == 'inference':
        return _inference(state, payload)
    raise ValueError(f'Unknown replicated execution operation: {operation}')
