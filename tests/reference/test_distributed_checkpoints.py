"""Complete rank state and failure boundaries using fresh two-process CPU groups."""
from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

import numpy as np
import pytest
import torch
import torch.distributed as dist

from hypergan.checkpoints import capture_rng, trainer_state
from hypergan.config import resolve_config
from hypergan.distributed_checkpoints import (
    save_distributed_checkpoint, restore_distributed_checkpoint,
)


class OrderedData:
    """Tiny local shuffled dataset; explicit state, all three global RNG consumers."""
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
        noise = .01 * torch.rand(batch_size, generator=generator)
        return {'real': torch.stack((values + noise, values.square()), dim=1)}

    def state_dict(self):
        return {'order': list(self.order), 'cursor': self.cursor, 'epoch': self.epoch, 'owner_rank': dist.get_rank()}

    def load_state_dict(self, state):
        if state['owner_rank'] != dist.get_rank():
            raise ValueError('Rank-specific sampler state was assigned to the wrong owner')
        self.order, self.cursor, self.epoch = list(state['order']), state['cursor'], state['epoch']

    def resume_identity(self):
        # Returning the same identity may still consume randomness. Metadata must
        # not change the numerical continuation or the RNG captured in a snapshot.
        torch.rand(1)
        random.random()
        np.random.random()
        return {'dataset': 'seven-ordered-points', 'content': Path(self.identity_file).read_text()}


class TinyImageGenerator(torch.nn.Module):
    """Original shape/recovery fixture, not a copied or qualified image GAN."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6)

    def forward(self, x):
        value = self.linear(x)
        noise = .001 * (torch.rand_like(value) + random.random() + float(np.random.random()))
        return (value + noise).tanh().reshape(len(x), 1, 2, 3)


class TinyImageDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 1)

    def forward(self, x):
        return self.linear(x.flatten(1))


def config(root):
    image = (root / 'images').exists()
    data = {'factory': 'image_folder', 'args': {'root': str(root / 'images'), 'height': 2, 'width': 3, 'mode': 'L', 'labels': True}} if image else {'factory': '__main__:OrderedData', 'args': {'identity_file': str(root / 'data-identity')}}
    components = {
        'generator': {'factory': '__main__:TinyImageGenerator', 'inputs': {'x': 'latent'}},
        'discriminator': {'factory': '__main__:TinyImageDiscriminator', 'inputs': {'x': 'candidate'}},
    } if image else {
        'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
        'discriminator': {'factory': 'mlp', 'args': {'input_dim': 2, 'output_dim': 1, 'hidden': [8]}, 'inputs': {'x': 'candidate'}},
    }
    return resolve_config({'data': data, 'components': components,
                           'prior': {'args': {'num_particles': 8, 'z_dim': 4}},
                           'gradient_penalty': {'lazy_k': 2},
                           'training': {'steps': 4, 'batch_size': 4, 'seed': 47}})


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def worker(mode, rank, rendezvous, root, run, output):
    from hypergan.distributed_training import ReplicatedCPUTrainer
    import hypergan.distributed_checkpoints as checkpoints
    root, run = Path(root), Path(run)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=3 if mode.startswith('missing') else 15))
    try:
        trainer = ReplicatedCPUTrainer(config(root), world_size=2)
        last = None
        result = {'passed': True}
        if mode in ('full', 'split', 'resume'):
            if mode == 'resume':
                _, info, last = restore_distributed_checkpoint(run, trainer, {'run_id': 'fixture-run'})
                assert trainer.step == 2 and info['step'] == 2
            for _ in range((4 if mode == 'full' else 2)):
                torch.rand(rank + 1)
                random.random()
                np.random.random(rank + 1)
                _, last = trainer.update()
            before = capture_rng()
            path = save_distributed_checkpoint(run, trainer, last,
                       {'run_id': 'fixture-run', 'attempt_id': 'attempt-two' if mode == 'resume' else 'attempt-one', 'next_sample_sequence': 5})
            _assert_equal(capture_rng(), before)
            torch.save(trainer_state(trainer, last), root / f'{mode}-rank{rank}.pt')
            result['checkpoint'] = str(path)
        elif mode.startswith('restore-'):
            before = copy.deepcopy(trainer_state(trainer, None))
            if mode == 'restore-mutating-hook':
                def mutate_input(self, state):
                    state['cursor'] += 1
                    OrderedData.load_state_dict(self, state)
                trainer.data.load_state_dict = types.MethodType(mutate_input, trainer.data)
            if mode == 'restore-live-failure' and rank == 1:
                live_data = trainer.data
                def fail_live_only(self, state):
                    if self is live_data:
                        raise RuntimeError('injected live-only data restore failure')
                    OrderedData.load_state_dict(self, state)
                trainer.data.load_state_dict = types.MethodType(fail_live_only, trainer.data)
            if mode == 'restore-not-ready' and rank == 1:
                trainer.checkpoint_ready = False
            if mode == 'restore-poisoned' and rank == 1:
                trainer._poisoned = True
            if mode == 'restore-topology':
                trainer.strategy_info['gradient_reduction'] = 'wrong-reducer'
            if mode == 'restore-config':
                trainer.config['training']['steps'] = 5
            if mode == 'restore-threads':
                torch.set_num_threads(2)
            if mode == 'restore-runtime':
                actual = checkpoints.runtime_info
                checkpoints.runtime_info = lambda: dict(actual(), torch='different-version')
            try:
                restore_distributed_checkpoint(run, trainer, {'run_id': 'fixture-run'})
            except (ValueError, RuntimeError) as exc:
                result['error'] = str(exc)
            else:
                raise AssertionError('Invalid restore unexpectedly succeeded')
            if mode == 'restore-live-failure':
                assert not trainer.checkpoint_ready and trainer._poisoned
                try:
                    restore_distributed_checkpoint(run, trainer, {'run_id': 'fixture-run'})
                except ValueError as exc:
                    assert 'fresh trainer' in str(exc)
                else:
                    raise AssertionError('Partially restored trainer reuse was accepted')
            else:
                _assert_equal(trainer_state(trainer, None), before)
        else:
            _, last = trainer.update()
            good = save_distributed_checkpoint(run, trainer, last, {'run_id': 'fixture-run', 'attempt_id': 'attempt-one'})
            pointer = (run / 'distributed-checkpoints' / 'latest.json').read_bytes()
            if mode == 'half':
                if rank == 1:
                    def fail_step():
                        raise RuntimeError('injected G optimizer failure after D')
                    trainer.opt_g.step = fail_step
                try:
                    trainer.update()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('Injected half-update did not fail')
                assert not trainer.checkpoint_ready
            elif mode == 'divergent' and rank == 1:
                next(iter(trainer.opt_d.state.values()))['exp_avg'].add_(1)
            elif mode == 'state-failure' and rank == 1:
                def fail_state():
                    raise RuntimeError('injected rank snapshot failure')
                trainer.data.state_dict = fail_state
            elif mode == 'identity-type':
                trainer.data.resume_identity = lambda: {'marker': True if rank else 1}
            elif mode == 'stage-failure' and rank == 0:
                actual = checkpoints.atomic_json
                def fail_manifest(path, value):
                    if Path(path).name == 'manifest.json':
                        raise OSError('injected rank-zero staging disk failure')
                    return actual(path, value)
                checkpoints.atomic_json = fail_manifest
            elif mode == 'metadata-bound':
                checkpoints.MAX_METADATA_BYTES = 256
            elif mode == 'missing-after-gather' and rank == 1:
                original_gather = dist.gather_object
                def exit_after_transfer(*args, **kwargs):
                    original_gather(*args, **kwargs)
                    os._exit(17)
                dist.gather_object = exit_after_transfer
            elif mode == 'missing' and rank == 1:
                os._exit(17)
            start = time.monotonic()
            try:
                save_distributed_checkpoint(run, trainer, last, {'run_id': 'fixture-run', 'attempt_id': 'attempt-two'})
            except (ValueError, RuntimeError) as exc:
                result.update(error=str(exc), elapsed=time.monotonic() - start)
            else:
                raise AssertionError('Invalid distributed checkpoint unexpectedly succeeded')
            assert (run / 'distributed-checkpoints' / 'latest.json').read_bytes() == pointer
            assert (good / 'manifest.json').is_file()
        Path(output).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def launch(root, run, mode):
    root.mkdir(exist_ok=True)
    run.mkdir(exist_ok=True)
    identity = root / 'data-identity'
    if not identity.exists():
        identity.write_text('original-v1')
    rendezvous = root / f'rendezvous-{mode}-{time.monotonic_ns()}'
    processes = []
    deadline = time.monotonic() + 45
    outputs = [root / f'{mode}-rank{rank}.json' for rank in range(2)]
    try:
        for rank in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, *(['-I'] if sys.flags.isolated else []), str(Path(__file__).resolve()), mode, str(rank),
                 str(rendezvous), str(root), str(run), str(outputs[rank])],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for rank, process in enumerate(processes):
            stdout, stderr = process.communicate(timeout=max(.1, deadline - time.monotonic()))
            expected = 17 if mode.startswith('missing') and rank == 1 else 0
            assert process.returncode == expected, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)
    return [json.loads(path.read_text()) for path in outputs if path.exists()]


@pytest.mark.parametrize('dataset', ['ordered', 'image'])
def test_fresh_group_resume_preserves_every_rank_rng_sampler_and_optimizer(tmp_path, dataset):
    root = tmp_path / 'fixture'
    root.mkdir()
    if dataset == 'image':
        from PIL import Image
        for index in range(7):
            path = root / 'images' / ('cats' if index % 2 else 'zebras') / f'{index}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('L', (3, 2), index * 40).save(path)
    launch(root, tmp_path / 'full', 'full')
    launch(root, tmp_path / 'split', 'split')
    launch(root, tmp_path / 'split', 'resume')
    states = []
    for rank in range(2):
        expected = torch.load(root / f'full-rank{rank}.pt', weights_only=True)
        actual = torch.load(root / f'resume-rank{rank}.pt', weights_only=True)
        _assert_equal(actual, expected)
        assert actual['step'] == 4 and actual['data']['epoch'] >= 2
        states.append(actual)
    assert not torch.equal(states[0]['streams']['prior'], states[1]['streams']['prior'])
    assert not torch.equal(states[0]['rng']['torch'], states[1]['rng']['torch'])
    if dataset == 'ordered':
        assert states[0]['data']['owner_rank'] == 0 and states[1]['data']['owner_rank'] == 1
    else:
        assert states[0]['last_batch']['real'].shape == (2, 1, 2, 3)
        assert states[0]['last_batch']['labels'].dtype == torch.int64
    generations = list((tmp_path / 'split' / 'distributed-checkpoints').glob('attempt-*'))
    assert len(generations) == 2
    assert all((path / 'rank-00000.pt').is_file() and (path / 'rank-00001.pt').is_file() for path in generations)


@pytest.mark.parametrize('mode', ['half', 'divergent', 'state-failure', 'identity-type', 'stage-failure', 'metadata-bound', 'missing', 'missing-after-gather'])
def test_failed_rank_or_staging_never_advances_last_complete_checkpoint(tmp_path, mode):
    results = launch(tmp_path / 'fixture', tmp_path / 'run', mode)
    assert results and all(result['error'] for result in results)
    if not mode.startswith('missing'):
        assert results[0]['error'] == results[1]['error']
    else:
        assert len(results) == 1 and results[0]['elapsed'] < 8
    assert not list((tmp_path / 'run' / 'distributed-checkpoints').glob('.pending-*'))
    assert not list((tmp_path / 'run' / 'distributed-checkpoints' / '.prepared').rglob('command-*'))


@pytest.mark.parametrize('mode', ['restore-config', 'restore-topology', 'restore-runtime', 'restore-threads', 'restore-poisoned', 'restore-not-ready', 'restore-data', 'restore-missing-rank', 'restore-pointer', 'restore-mutating-hook', 'restore-live-failure'])
def test_incompatible_or_incomplete_checkpoint_rejected_before_live_mutation(tmp_path, mode):
    root, run = tmp_path / 'fixture', tmp_path / 'run'
    saved = launch(root, run, 'split')
    generation = Path(saved[0]['checkpoint'])
    if mode == 'restore-data':
        (root / 'data-identity').write_text('changed-content-or-class-map')
    if mode == 'restore-missing-rank':
        (generation / 'rank-00001.pt').unlink()
    if mode == 'restore-pointer':
        path = run / 'distributed-checkpoints' / 'latest.json'
        value = json.loads(path.read_text())
        value['step'] = 1
        path.write_text(json.dumps(value))
    results = launch(root, run, mode)
    assert results[0]['error'] == results[1]['error']


def test_requires_explicit_initialized_group():
    with pytest.raises(RuntimeError, match='Initialize'):
        save_distributed_checkpoint('.', None, None, {})


if __name__ == '__main__':
    worker(sys.argv[1], int(sys.argv[2]), *sys.argv[3:])
