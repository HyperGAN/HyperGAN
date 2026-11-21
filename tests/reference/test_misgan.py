"""Two ParticleGAN critics with one learned MoG and one joint generator update."""
import json
from unittest.mock import patch

import pytest
import torch

import hypergan as hg
from hypergan.checkpoints import read_checkpoint
from hypergan.training import ReferenceTrainer, resume, train
from tests.reference.test_particlegan_integration import equal


class MaskedRows:
    resume_stateless = True

    def __call__(self, n, *, generator):
        data = torch.randn(n, 585, generator=generator)
        mask = (torch.rand(n, 585, generator=generator) > 0.3).float()
        return {'real': torch.cat((data * mask, mask), dim=1), 'mask': mask}


class MaskedPair(torch.nn.Module):
    def forward(self, data, mask):
        return torch.cat((data * mask, mask), dim=1)


@pytest.fixture
def model(tmp_path):
    def linear(role, width, output, source):
        return {'role': role, 'source': 'linear()', 'input_shape': ['B', width],
                'output_shape': ['B', output], 'inputs': {'x': source}}

    spec = {
        'format': 'hypergan-model/1', 'name': 'test/misgan', 'defaults': 'particlegan',
        'data': {'factory': f'{__name__}:MaskedRows'},
        'prior': {'kind': 'mog', 'z_dim': 4, 'num_particles': 32, 'fixed_sigma': 0.05},
        'networks': {
            'gx': linear('generator', 4, 585, 'latent'), 'gm': linear('auxiliary', 4, 585, 'latent'),
            'pair': {'role': 'auxiliary', 'module': f'{__name__}:MaskedPair',
                     'inputs': {'data': 'gx', 'mask': 'gm'}},
            'dx': linear('critic', 1170, 1, 'candidate'), 'dm': linear('critic', 585, 1, 'candidate'),
        },
        'losses': [
            hg.adversarial('dx', fake='pair', critic_weight=1, generator_weight=0.2, penalty=0.25),
            hg.adversarial('dm', real='batch.mask', fake='gm', penalty=0.75),
            {'type': 'prior', 'weight': 1.0},
        ],
        'train': {'steps': 3, 'batch_size': 8, 'device': 'cpu'},
        'metrics': {'every_steps': 1},
        'checkpoints': {'latest_only': True},
    }
    path = tmp_path / 'misgan.toml'
    hg.save(spec, path)
    return hg.load(path)


def test_independent_critic_updates_and_joint_generator_objective(model):
    def update(model):
        trainer = ReferenceTrainer(hg.resolve(model))
        dx, dm = trainer.program.adversarial_terms
        opt_x, opt_m = trainer.critic_optimizers.values()
        assert dx.gan is not dm.gan and dx.penalty_fn is not dm.penalty_fn
        assert dx.gan.record is opt_x.record and dm.gan.record is opt_m.record
        assert opt_x.record is not opt_m.record
        assert dx.critic_weight == dm.critic_weight == dm.generator_weight == 1
        assert dx.penalty_fn.regularizer.coeff == 0.25 and dm.penalty_fn.regularizer.coeff == 0.75
        g_terms = []
        def watch(loss):
            def generator_loss(*scores):
                assert all(not p.requires_grad for p in trainer.program.critic_parameters)
                value = loss(*scores)
                g_terms.append(value.detach())
                return value
            return generator_loss

        with (patch.object(dx.gan, 'g_loss', watch(dx.gan.g_loss)),
              patch.object(dm.gan, 'g_loss', watch(dm.gan.g_loss)),
              patch.object(opt_x, 'step', wraps=opt_x.step) as step_x,
              patch.object(opt_m, 'step', wraps=opt_m.step) as step_m,
              patch.object(trainer.opt_g, 'step', wraps=trainer.opt_g.step) as step_g,
              patch.object(trainer.spread, 'forward', wraps=trainer.spread.forward) as reg,
              patch.object(trainer.prior, 'sample', wraps=trainer.prior.sample) as draw):
            prior_before = trainer.prior.z.detach().clone()
            row, _ = trainer.update()
        assert step_x.call_count == step_m.call_count == step_g.call_count == reg.call_count == draw.call_count == 1
        assert not torch.equal(trainer.prior.z, prior_before)
        assert len(g_terms) == 2
        assert row['g_adversarial_weighted'] == (dx.generator_weight * g_terms[0] + g_terms[1]).item()
        assert row['g_loss'] == pytest.approx(row['g_adversarial_weighted'] + row['prior_loss'])
        return row, [(term.module.state_dict(), opt.state_dict()) for term, opt in zip((dx, dm), (opt_x, opt_m))]

    weighted, critics = update(model)
    baseline, other_critics = update(hg.override(model, {'losses.0.generator_weight': 1.0}))
    assert baseline['d_loss'] == weighted['d_loss']
    assert baseline['g_adversarial_weighted'] != weighted['g_adversarial_weighted']
    # The generator phase can publish observations to critic records. Compare
    # the networks and the critic optimizer moments produced by the critic phase.
    for (network, opt), (other_network, other_opt) in zip(critics, other_critics):
        equal(network, other_network)
        equal(opt['state'], other_opt['state'])


def test_latest_only_exact_resume_after_failed_write(model, tmp_path, monkeypatch):
    run, full = tmp_path / 'resumed', tmp_path / 'full'
    train(model.path, full, checkpoint_every=1)
    original_save = torch.save

    def fail_step_two(state, destination, *args, **kwargs):
        if isinstance(state, dict) and state.get('step') == 2 and 'optimizers' in state:
            destination.write(b'partial')
            raise OSError('injected step 2 checkpoint failure')
        return original_save(state, destination, *args, **kwargs)

    def retained(step):
        path, metadata, state = read_checkpoint(run)
        assert state['step'] == metadata['step'] == step
        assert list((run / 'checkpoints').glob('*-step-*')) == [path]
        return state

    with monkeypatch.context() as fault:
        fault.setattr(torch, 'save', fail_step_two)
        with pytest.raises(OSError, match='injected step 2'):
            train(model.path, run, checkpoint_every=1)
    retained(1)
    prefix = (run / 'events.jsonl').read_bytes()
    resume(run, stop_after_steps=1)
    retained(2)
    assert (run / 'events.jsonl').read_bytes().startswith(prefix)
    resume(run)
    equal(retained(3), read_checkpoint(full)[2])
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    assert {e['step'] for e in events if e['event'] == 'checkpoint'} == {0, 1, 2, 3}
