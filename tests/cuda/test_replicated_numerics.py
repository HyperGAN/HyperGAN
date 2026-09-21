"""Opt-in complete two-GPU numerical oracles; all hardware gates are mandatory."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
# Direct worker entry points need the repository fixture package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.hndl_fixtures import fixture_linear, fixture_network
import torch.distributed as dist
from torch import nn

from hypergan.config import config_values, resolve_config
from hypergan.distributed import Collectives
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.training import ReferenceTrainer


class ConditionalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = fixture_linear(6, 2, bias=False)

    def forward(self, x, condition):
        return self.linear(torch.cat((x, condition), dim=1))


class CubicCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = fixture_network('cubic', (2,), (1,))
        with torch.no_grad():
            self.network.nodes.n_projection.weight.copy_(torch.tensor([[1.4, 1.1]]))

    def forward(self, x):
        # Trusted custom code can change the ambient device. Every following
        # NCCL metadata exchange must rebind its rank-owned GPU.
        torch.cuda.set_device(1 - dist.get_rank())
        return self.network(x)


def _config(mode):
    return resolve_config({
        'training': {'device': 'cuda', 'steps': 3, 'batch_size': 8, 'seed': 817, 'lr_anneal_start': .3},
        'prior': {'kind': 'mog', 'args': {'num_particles': 12, 'z_dim': 4, 'sigma_rel': .03, 'device': 'cuda'}},
        'adversarial': {'mode': mode},
        'gradient_penalty': {'lazy_k': 2, 'kappa': .01},
        'components': {
            'encoder': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 2, 'bias': False},
                        'inputs': {'input': 'batch.condition'}},
            'generator': {'factory': f'{__name__}:ConditionalGenerator',
                          'inputs': {'x': 'latent', 'condition': 'components.encoder'}},
            # No additive critic bias: RA has an exact null direction there,
            # whose reduction roundoff Adam can amplify (not objective error).
            'discriminator': {'factory': f'{__name__}:CubicCritic', 'inputs': {'x': 'candidate'}},
        },
        'objectives': [{'factory': 'mse', 'inputs': {'input': 'generated', 'target': 'batch.real'},
                        'weight': .7, 'detach': ['target']}],
    })


def _draw(trainer, step, rank=None):
    condition = torch.tensor([[-1.4, .2], [-.7, .8], [-1.1, -.4], [-.3, 1.2],
                              [.2, -.8], [.9, .4], [1.3, -.6], [.6, 1.4]], device=trainer.device)
    real = condition * (1.2 + step * .1) + .15
    # Same IDs repeat across ranks AND microbatches, so covariance must use the
    # global union once. Standardized MoG also gives unsampled rows gradients.
    ids = torch.tensor([0, 0, 2, 4, 2, 6, 6, 0], device=trainer.device)
    eps = torch.linspace(-.4, .7, 32, device=trainer.device).reshape(8, 4) + step * .02
    if rank is not None:
        condition, real, ids, eps = (value.chunk(2)[rank] for value in (condition, real, ids, eps))
    return {'real': real, 'condition': condition}, (trainer.prior(ids, eps=eps), ids)


def _state(trainer):
    return {'graph': trainer.graph.state_dict(), 'prior': trainer.prior.state_dict(),
            'ema_graph': trainer.ema_graph.state_dict(), 'ema_prior': trainer.ema_prior.state_dict(),
            'opt_g': trainer.opt_g.state_dict(), 'opt_d': trainer.opt_d.state_dict(),
            'base_lrs': trainer.base_lrs, 'step': trainer.step}


def _close(actual, expected, path='state'):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6,
                                   msg=lambda message: f'{path}: {message}')
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _close(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            _close(a, b, f'{path}[{index}]')
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=3e-5, abs=3e-6), path
    else:
        assert actual == expected, path


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=20), device_id=torch.device('cuda', rank))
    try:
        # Real primitive double backward uses NCCL reduce-scatter, rather than
        # relying solely on the separately validated diagnostic implementation.
        collectives = Collectives(2)
        x = torch.tensor([[rank + 1.]], device=collectives.device, requires_grad=True)
        loss = collectives.mean(x).square().sum()
        first, = torch.autograd.grad(loss, x, create_graph=True)
        second, = torch.autograd.grad(first.sum(), x)
        torch.testing.assert_close(first, torch.full_like(x, 3.), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.full_like(x, 2.), rtol=0, atol=0)
        for mode in ('rp', 'ra'):
            for accumulation in (1, 2):
                config = _config(mode)
                replica = ReplicatedTrainer(config, accumulation_steps=accumulation)
                local = config_values(config)
                local['training']['device'] = f'cuda:{rank}'
                local['prior']['args']['device'] = f'cuda:{rank}'
                reference = ReferenceTrainer(resolve_config(local))
                for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
                    getattr(reference, name).load_state_dict(getattr(replica, name).state_dict())
                initial_encoder = copy.deepcopy(replica.graph.models['encoder'].state_dict())
                assert replica.config['training']['device'] == 'cuda'
                assert replica.strategy_info['name'] == 'cuda-replicated-nccl'
                assert replica.device == torch.device('cuda', rank)
                for step in range(1, 4):
                    batch, draw = _draw(replica, step, rank)
                    actual, _ = replica.update(batch, draw)
                    full, global_draw = _draw(reference, step)
                    expected, _ = reference.update(full, global_draw)
                    context = f'{mode}/A{accumulation}/step{step}/rank{rank}'
                    _close(_state(replica), _state(reference), context)
                    for key in ('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty'):
                        _close(actual[key], expected[key], f'{context}.{key}')
                    assert (actual['gradient_penalty'] > 0) == (step == 2)
                    assert replica.checkpoint_ready and torch.cuda.current_device() == rank
                assert any(not torch.equal(initial_encoder[key], value)
                           for key, value in replica.graph.models['encoder'].state_dict().items())
                assert replica.opt_g.state[replica.prior.z]['exp_avg'][[1, 3, 5, 7, 8, 9, 10, 11]].abs().sum() > 0
        # Real default recipe, actual automatic sampler/prior, global sharding,
        # reducer-presence tensor and metrics must all use valid NCCL devices.
        default = resolve_config({'training': {'device': 'cuda', 'steps': 2},
                                  'prior': {'args': {'num_particles': 32, 'z_dim': 4}}})
        trainer = ReplicatedTrainer(default, accumulation_steps=2)
        for _ in range(2):
            metrics, batch = trainer.update()
            assert trainer.checkpoint_ready and batch['real'].device == trainer.device
            assert all(torch.isfinite(torch.tensor(metrics[key])) for key in ('d_loss', 'g_loss'))
        parameters = trainer._parameters(trainer.opt_g)
        for parameter in parameters:
            parameter.grad = None
        parameters[0].grad = None if rank == 0 else torch.ones_like(parameters[0])
        trainer._reduce_gradients(trainer.opt_g, 'partial local participation')
        assert torch.equal(parameters[0].grad, torch.full_like(parameters[0], .5))
        assert all(parameter.grad is None for parameter in parameters[1:])
        Path(output).write_text(json.dumps({'passed': True, 'rank': rank}))
    finally:
        dist.destroy_process_group()


def test_two_gpu_complete_updates_match_global_reference(tmp_path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'This acceptance gate requires two visible CUDA GPUs'
    assert dist.is_nccl_available(), 'This acceptance gate requires NCCL'
    children, logs = [], []
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC='1000')
    try:
        for rank in range(2):
            log = (tmp_path / f'rank-{rank}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []),
                str(Path(__file__).resolve()), '--worker', str(rank), str(tmp_path / 'rendezvous'),
                str(tmp_path / f'rank-{rank}.json')], stdout=log, stderr=subprocess.STDOUT, env=env))
        deadline = time.monotonic() + 120
        for child in children:
            child.wait(timeout=max(.1, deadline - time.monotonic()))
        for log in logs:
            log.flush()
        errors = '\n'.join(path.read_text() for path in sorted(tmp_path.glob('*.log')))
        assert all(child.returncode == 0 for child in children), errors
        assert all(json.loads((tmp_path / f'rank-{rank}.json').read_text())['passed'] for rank in range(2))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == '__main__':
    assert sys.argv[1] == '--worker'
    _worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
