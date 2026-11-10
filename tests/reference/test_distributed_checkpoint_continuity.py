"""Distributed checkpoint continuity across aggregation strategies, cadences and transports.

This is the one test file for distributed checkpoint continuity (Issue #186). Each
matrix case trains an uninterrupted reference and a split run that saves at the
split step in one launch, then resumes in a fresh process group, and requires the
resumed trajectory to equal the uninterrupted one bit for bit. ``tcp`` cases run
as two single-rank nodes and resume on a fresh port. The ``heavy`` tests start one
worker process per rank, this file re-executing itself as the worker.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import pytest
import torch
import torch.distributed as dist

from hypergan.checkpoints import capture_rng, restore_rng, trainer_state
from hypergan.config import resolve_config
from hypergan.distributed_checkpoints import (
    _validate_rank_rng, _validate_restore_identity, restore_distributed_checkpoint, save_distributed_checkpoint,
)
from hypergan.recipes import move_tensors


class OrderedData:
    """Deterministic shuffled dataset whose sampler state is owned by one rank."""

    def __init__(self, identity_file):
        self.identity_file = identity_file
        self.order, self.cursor, self.epoch = [], 0, 0

    def __call__(self, batch_size, *, generator):
        selected = []
        for _ in range(batch_size):
            if self.cursor == len(self.order):
                self.order = torch.randperm(7, generator=generator).tolist()
                self.cursor = 0
                self.epoch += 1
            selected.append(self.order[self.cursor])
            self.cursor += 1
        values = torch.tensor(selected, dtype=torch.float32) / 7
        noise = 0.01 * torch.rand(batch_size, generator=generator)
        return {'real': torch.stack((values + noise, values.square()), dim=1)}

    def state_dict(self):
        return {'order': list(self.order), 'cursor': self.cursor, 'epoch': self.epoch, 'owner_rank': dist.get_rank()}

    def load_state_dict(self, state):
        if state['owner_rank'] != dist.get_rank():
            raise ValueError('Rank-specific sampler state was assigned to the wrong owner')
        self.order, self.cursor, self.epoch = list(state['order']), state['cursor'], state['epoch']

    def resume_identity(self):
        return {'dataset': 'seven-ordered-points', 'content': Path(self.identity_file).read_text()}


def _config(root, steps):
    return resolve_config({
        'data': {'factory': '__main__:OrderedData', 'args': {'identity_file': str(Path(root) / 'data-identity')}},
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'mlp', 'args': {'input_dim': 2, 'output_dim': 1, 'hidden': [8]}, 'inputs': {'x': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 8, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2},
        'training': {'steps': steps, 'batch_size': 4, 'seed': 47},
    })


def _aggregation(strategy, cadence, outer_lr=1.0, outer_momentum=0.0):
    return {'strategy': strategy, 'sync_every_steps': cadence, 'outer_lr': outer_lr, 'outer_momentum': outer_momentum}


# ==============================================================================
# Rank workers
# ==============================================================================

def _trainer(transport, rank, rendezvous, root, steps, aggregation):
    from hypergan.distributed_training import ReplicatedCPUTrainer
    topology = {} if transport == 'file' else {'nodes': 2, 'node_rank': rank, 'ranks_per_node': 1, 'rendezvous': rendezvous}
    return ReplicatedCPUTrainer(_config(root, steps), world_size=2, aggregation=aggregation, **topology)


def _snapshot(trainer, last_batch, metrics=()):
    # trainer_state shares storage with the live modules; keep this step's values.
    state = copy.deepcopy(trainer_state(trainer, last_batch))
    state['aggregation'] = move_tensors(trainer.aggregation_strategy.state_dict(), 'cpu')
    state['metrics'] = list(metrics)
    return state


def _train(trainer, steps, metrics, last_batch=None):
    for _ in range(steps):
        row, last_batch = trainer.update()
        metrics.append({key: row[key] for key in ('step', 'd_loss', 'g_loss')})
    return last_batch


def _worker(mode, transport, rank, rendezvous, root, run, output, aggregation, steps, split):
    rank, steps, split, aggregation = int(rank), int(steps), int(split), json.loads(aggregation)
    root, run = Path(root), Path(run)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        result = {}
        metadata = {'run_id': 'continuity-run', 'attempt_id': 'attempt-one', 'next_sample_sequence': split + 1}
        if mode == 'reference':
            # The uninterrupted run and the run that saves at the split step.
            trainer, metrics = _trainer(transport, rank, rendezvous, root, steps, aggregation), []
            last = _train(trainer, split, metrics)
            at_split = _snapshot(trainer, last, metrics)
            last = _train(trainer, steps - split, metrics, last)
            split_trainer, split_metrics = _trainer(transport, rank, rendezvous, root, steps, aggregation), []
            split_last = _train(split_trainer, split, split_metrics)
            result['checkpoint'] = str(save_distributed_checkpoint(run, split_trainer, split_last, metadata))
            torch.save({'at_split': at_split, 'final': _snapshot(trainer, last, metrics),
                        'split': _snapshot(split_trainer, split_last, split_metrics)}, root / f'reference-rank{rank}.pt')
        elif mode == 'resume':
            trainer = _trainer(transport, rank, rendezvous, root, steps, aggregation)
            _, info, last = restore_distributed_checkpoint(run, trainer, {'run_id': 'continuity-run'})
            result['restored_step'], restored = info['step'], _snapshot(trainer, last)
            metrics = []
            last = _train(trainer, steps - split, metrics, last)
            torch.save({'restored': restored, 'final': _snapshot(trainer, last, metrics)}, root / f'resume-rank{rank}.pt')
        elif mode == 'non_cadence':
            # A checkpoint between synchronizations first synchronizes once (review 2026-10-25 H3).
            trainer = _trainer(transport, rank, rendezvous, root, steps, aggregation)
            last = _train(trainer, 1, [])
            try:
                save_distributed_checkpoint(run, trainer, last, dict(metadata, next_sample_sequence=2))
                result['rejected'] = None
            except ValueError as exc:
                result['rejected'] = str(exc)
            result.update(synced_step=trainer.aggregation_strategy.synced_step, checkpoint_ready=trainer.checkpoint_ready,
                          weights=trainer._digest(trainer.graph.state_dict()))
        elif mode == 'step0':
            trainer = _trainer(transport, rank, rendezvous, root, steps, aggregation)
            assert trainer.step == 0 and trainer.checkpoint_ready
            save_distributed_checkpoint(run, trainer, None, dict(metadata, next_sample_sequence=1))
            fresh = _trainer(transport, rank, rendezvous, root, steps, aggregation)
            _, info, last = restore_distributed_checkpoint(run, fresh, {'run_id': 'continuity-run'})
            assert (fresh.step, info['step'], last, fresh.checkpoint_ready) == (0, 0, None, True)
            for key, value in trainer.graph.state_dict().items():
                assert torch.equal(value, fresh.graph.state_dict()[key]), key
        Path(output).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _launch(tmp_path, mode, transport, aggregation, steps=4, split=2, timeout=120):
    root, run = tmp_path / 'root', tmp_path / 'run'
    root.mkdir(exist_ok=True)
    run.mkdir(exist_ok=True)
    identity = root / 'data-identity'
    if not identity.exists():
        identity.write_text('continuity-v1')
    stamp = time.monotonic_ns()
    rendezvous = (root / f'rendezvous-{mode}-{stamp}').as_uri() if transport == 'file' else f'tcp://127.0.0.1:{_free_port()}'
    outputs = [root / f'{mode}-{stamp}-rank{rank}.json' for rank in range(2)]
    command = [sys.executable, *(['-I'] if sys.flags.isolated else []), str(Path(__file__).resolve())]
    processes = [subprocess.Popen([*command, mode, transport, str(rank), rendezvous, str(root), str(run), str(outputs[rank]),
                                   json.dumps(aggregation), str(steps), str(split)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for rank in range(2)]
    failures, deadline = [], time.monotonic() + timeout
    try:
        for rank, process in enumerate(processes):
            _, stderr = process.communicate(timeout=max(0.1, deadline - time.monotonic()))
            if process.returncode:
                failures.append(f'rank {rank} exited {process.returncode}:\n{stderr}')
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert not failures, '\n'.join(failures)
    return [json.loads(path.read_text()) for path in outputs], root, run


def _assert_identical(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype, path
        assert torch.equal(left, right), f'{path}: tensors differ'
    elif isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), f'{path}: keys {set(left) ^ set(right)}'
        for key in left:
            _assert_identical(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), f'{path}: length'
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_identical(a, b, f'{path}[{index}]')
    else:
        assert left == right, f'{path}: {left!r} != {right!r}'


def _without(state, *keys):
    return {key: value for key, value in state.items() if key not in keys}


# ==============================================================================
# Save -> fresh group -> resume continuity matrix
# ==============================================================================

MATRIX = [  # strategy, cadence, outer_lr, outer_momentum, transport, steps, split
    ('gradient_allreduce', 1, 1.0, 0.0, 'file', 4, 2),
    ('parameter_average', 1, 1.0, 0.0, 'file', 4, 2),
    ('parameter_average', 3, 1.0, 0.0, 'file', 6, 3),
    ('pseudo_gradient', 1, 0.5, 0.8, 'file', 4, 2),
    ('pseudo_gradient', 3, 0.6, 0.9, 'file', 6, 3),
    ('gradient_allreduce', 1, 1.0, 0.0, 'tcp', 4, 2),
    ('parameter_average', 2, 1.0, 0.0, 'tcp', 4, 2),
    ('pseudo_gradient', 2, 0.7, 0.85, 'tcp', 4, 2),
]


@pytest.mark.heavy
@pytest.mark.parametrize('strategy,cadence,outer_lr,outer_momentum,transport,steps,split', MATRIX,
                         ids=[f'{case[0]}-every{case[1]}-{case[4]}' for case in MATRIX])
def test_resumed_trajectory_equals_the_uninterrupted_one(tmp_path, strategy, cadence, outer_lr, outer_momentum,
                                                         transport, steps, split):
    aggregation = _aggregation(strategy, cadence, outer_lr, outer_momentum)
    reference, root, run = _launch(tmp_path, 'reference', transport, aggregation, steps, split)
    checkpoint = Path(reference[0]['checkpoint'])
    assert reference[1]['checkpoint'] == str(checkpoint)

    # The manifest names the strategy and a topology without rank-local or ephemeral fields.
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    expected_metadata = {'strategy': strategy, 'sync_every_steps': cadence}
    if strategy == 'pseudo_gradient':
        expected_metadata.update(outer_lr=outer_lr, outer_momentum=outer_momentum)
    assert manifest['step'] == split and manifest['aggregation'] == expected_metadata
    topology = manifest['identity']['topology']
    assert 'node_rank' not in topology and 'rendezvous' not in topology
    assert (topology['world_size'], topology['nodes'], topology['ranks_per_node'], topology['global_batch_size'],
            topology['local_batch_size']) == (2, 2 if transport == 'tcp' else 1, 1 if transport == 'tcp' else 2, 4, 2)
    payloads = [torch.load(checkpoint / f'rank-{rank:05d}.pt', weights_only=True) for rank in range(2)]
    for payload in payloads:
        saved = payload['aggregation']
        assert (saved['strategy'], saved['step_count'], saved['sync_every_steps']) == (strategy, split, cadence)
        if strategy == 'pseudo_gradient':
            # Outer state stays on the trainer device in memory but is stored on the CPU.
            assert saved['anchor'] and set(saved['anchor']) == set(saved['momentum_buffer'])
            assert {value.device.type for value in (*saved['anchor'].values(), *saved['momentum_buffer'].values())} == {'cpu'}
        assert isinstance(payload['optimizers'][0]['regularizer']['latent']['history'], torch.Tensor)
        assert 'record' in payload['optimizers'][1]['regularizer']

    resumed, _, _ = _launch(tmp_path, 'resume', transport, aggregation, steps, split)
    assert [rank['restored_step'] for rank in resumed] == [split, split]
    finals = []
    for rank in range(2):
        expected = torch.load(root / f'reference-rank{rank}.pt', weights_only=False)
        actual = torch.load(root / f'resume-rank{rank}.pt', weights_only=False)
        # The checkpoint restores exactly what the split run saved, RNG included, and
        # that equals the uninterrupted run at the split step.
        _assert_identical(_without(actual['restored'], 'rng', 'metrics'), _without(payloads[rank], 'rng'), f'rank {rank} restored')
        _assert_identical(actual['restored']['rng'], payloads[rank]['rng'], f'rank {rank} restored rng')
        _assert_identical(_without(expected['split'], 'rng', 'metrics'), _without(expected['at_split'], 'rng', 'metrics'),
                          f'rank {rank} split run')
        # The resumed run then reproduces the uninterrupted one exactly.
        _assert_identical(_without(actual['final'], 'rng', 'metrics'), _without(expected['final'], 'rng', 'metrics'),
                          f'rank {rank} final')
        assert actual['final']['metrics'] == expected['final']['metrics'][split:]
        assert actual['final']['aggregation']['step_count'] == steps
        finals.append(actual['final']['metrics'])
    assert finals[0] == finals[1], 'ranks report different global metrics'


@pytest.mark.heavy
@pytest.mark.parametrize('strategy', ['parameter_average', 'pseudo_gradient'])
def test_checkpoint_between_synchronizations_synchronizes_once(tmp_path, strategy):
    results, _, run = _launch(tmp_path, 'non_cadence', 'tcp', _aggregation(strategy, 2, 0.5, 0.8))
    for result in results:
        assert result['rejected'] is None, result['rejected']
        assert result['synced_step'] == 1 and result['checkpoint_ready'] is True
    assert results[0]['weights'] == results[1]['weights']
    assert json.loads((run / 'distributed-checkpoints' / 'latest.json').read_text())['step'] == 1


@pytest.mark.heavy
def test_step_zero_checkpoint_saves_and_restores(tmp_path):
    _, _, run = _launch(tmp_path, 'step0', 'file', _aggregation('parameter_average', 2))
    root = run / 'distributed-checkpoints'
    latest = json.loads((root / 'latest.json').read_text())
    manifest = json.loads((root / latest['checkpoint'] / 'manifest.json').read_text())
    assert latest['step'] == manifest['step'] == 0
    assert manifest['run_id'] == 'continuity-run' and len(manifest['ranks']) == 2


@pytest.mark.heavy
def test_product_run_resumes_from_its_latest_distributed_checkpoint(tmp_path):
    """run_train writes manifest, events and a distributed checkpoint; run_resume continues the step count."""
    from hypergan.replicated_execution import run_resume, run_train
    from tests.config_fixtures import write_default
    config = write_default(tmp_path / 'project', device='cpu', defaults='hypergan')
    run = tmp_path / 'runs' / 'replicated'
    profile = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 1}}
    run_train(config, run, profile=profile, checkpoint_every=1, stop_after_steps=2, preview_every=0)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] in ('stopped', 'complete') and (manifest['steps'], manifest['last_durable_step']) == (2, 2)
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    assert [event['step'] for event in events if event.get('event') == 'train'] == [1, 2]
    assert json.loads((run / 'distributed-checkpoints' / 'latest.json').read_text())['step'] == 2
    run_resume(run, profile=profile, checkpoint_every=1, stop_after_steps=2, preview_every=0)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert (manifest['steps'], manifest['last_durable_step']) == (4, 4)


# ==============================================================================
# Restore and publication validation (in process)
# ==============================================================================

def test_restore_rejects_a_different_cluster_topology(monkeypatch):
    import hypergan.distributed_checkpoints as checkpoints
    saved = {
        'hypergan_checkpoint_version': 1, 'config': {'training': {'batch_size': 8, 'steps': 10}}, 'config_sha256': 'dummy',
        'runtime': {'device': 'cpu', 'backend': 'gloo', 'world_size': 2, 'threads': 1, 'interop_threads': 1},
        'implementation': {}, 'data_contract': {'identity': {}},
        'topology': {'world_size': 2, 'nodes': 2, 'ranks_per_node': 1, 'global_batch_size': 8, 'local_batch_size': 4},
    }
    monkeypatch.setattr(checkpoints, 'fingerprint', lambda config: 'dummy')
    monkeypatch.setattr(checkpoints, 'resume_compatible', lambda left, right: True)
    monkeypatch.setattr(checkpoints, 'validate_checkpoint_compatibility', lambda state: None)
    monkeypatch.setattr(checkpoints, 'validate_runtime', lambda saved, current, **kwargs: None)
    monkeypatch.setattr(checkpoints, 'validate_implementation', lambda saved, current: None)
    _validate_restore_identity(saved, copy.deepcopy(saved))
    one_node = copy.deepcopy(saved)
    one_node['topology'].update(nodes=1, ranks_per_node=2)
    four_ranks = copy.deepcopy(saved)
    four_ranks['topology']['world_size'] = four_ranks['runtime']['world_size'] = 4
    for current in (one_node, four_ranks):
        with pytest.raises(ValueError, match='Distributed checkpoint config/data/topology identity differs'):
            _validate_restore_identity(saved, current)


def test_restore_rejects_missing_or_foreign_rng_state():
    class CUDATrainer:
        device, rank, world_size, ranks_per_node = torch.device('cuda:0'), 0, 2, 2

    for rng in (None, {'cuda_device': 1}, {'cuda_device': '0'}):
        with pytest.raises(ValueError, match='Checkpoint CUDA RNG current device differs from the owning rank'):
            _validate_rank_rng(CUDATrainer(), {'rng': rng})
    valid = capture_rng()
    for missing in ('torch', 'python', 'numpy'):
        with pytest.raises(KeyError):
            restore_rng({key: value for key, value in valid.items() if key != missing})


@pytest.mark.parametrize('aggregation,match', [
    ({'strategy': 123}, 'strategy must be a safe nonempty identifier'),
    ({'strategy': '../traversal'}, 'strategy must be a safe nonempty identifier'),
    ({'strategy': 'test', 'sync_every_steps': 0}, 'sync_every_steps must be a positive integer'),
    ({'strategy': 'test', 'sync_every_steps': -1}, 'sync_every_steps must be a positive integer'),
    ({'strategy': 'test', 'sync_every_steps': True}, 'sync_every_steps must be a positive integer'),
    ({'strategy': 'test', 'outer_lr': 0.0}, 'outer_lr must be a positive finite number'),
    ({'strategy': 'test', 'outer_lr': -0.5}, 'outer_lr must be a positive finite number'),
    ({'strategy': 'test', 'outer_lr': float('inf')}, 'outer_lr must be a positive finite number'),
    ({'strategy': 'test', 'outer_lr': True}, 'outer_lr must be a positive finite number'),
    ({'strategy': 'test', 'outer_momentum': 1.0}, r'outer_momentum must be in \[0.0, 1.0\)'),
    ({'strategy': 'test', 'outer_momentum': -0.1}, r'outer_momentum must be in \[0.0, 1.0\)'),
    ({'strategy': 'test', 'outer_momentum': float('nan')}, r'outer_momentum must be in \[0.0, 1.0\)'),
    ({'strategy': 'test', 'outer_momentum': False}, r'outer_momentum must be in \[0.0, 1.0\)'),
    ([], 'metadata must be a dictionary'),
    (None, None),
    ({'strategy': 'pseudo_gradient', 'sync_every_steps': 2, 'outer_lr': 0.5, 'outer_momentum': 0.9}, None),
])
def test_publication_validates_manifest_aggregation_metadata(tmp_path, aggregation, match):
    import hypergan.distributed_commit as commit
    identity = {'config': {'training': {'steps': 4}}, 'topology': {'world_size': 2}}
    controller, nonce = uuid.uuid4().hex, '1234567890ab'
    staging = commit.preparation_directory(tmp_path, 'attempt-1', controller, 1, nonce, create=True)
    ranks = []
    for rank in range(2):
        payload = f'rank {rank}'.encode()
        (staging / f'rank-{rank:05d}.pt').write_bytes(payload)
        ranks.append({'rank': rank, 'file': f'rank-{rank:05d}.pt', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    info = {'schema_version': 1, 'kind': commit.KIND, 'run_id': 'run-1', 'attempt_id': 'attempt-1', 'step': 2,
            'identity': identity, 'replicated_sha256': 'a' * 64, 'ranks': ranks, 'next_sample_sequence': 3}
    if aggregation is not None:
        info['aggregation'] = aggregation
    (staging / 'manifest.json').write_text(json.dumps(info))
    receipt = commit.make_prepared_receipt(tmp_path, staging, info, controller_id=controller, command_sequence=1, nonce=nonce)
    validate = lambda: commit._validate_prepared(tmp_path, receipt, run_id='run-1', attempt_id='attempt-1',
                                                 controller_id=controller, command_sequence=1, identity=identity)
    if match is None:
        assert validate()[2] == info
    else:
        with pytest.raises(ValueError, match='Prepared checkpoint aggregation ' + match):
            validate()


if __name__ == '__main__':
    _worker(*sys.argv[1:])
