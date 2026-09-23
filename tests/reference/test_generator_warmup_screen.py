"""Extra G steps preserve opponent/prior state and return to ordinary training."""
import copy
from pathlib import Path

import pytest
import torch
from hypergan.config import load_config, write_default
from hypergan.training import ReferenceTrainer
from hypergan.startup_dynamics import _same_state


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'research/startup_tuning'))
    import generator_warmup_screen as screen
    path = write_default(tmp_path / 'config', device='cpu')
    path.write_text(path.read_text().replace('steps = 5', 'steps = 512')
                    .replace('num_particles = 20000', 'num_particles = 32'))
    return screen, path


def test_extra_g_leaves_critic_prior_optimizer_and_clocks_fixed(setup):
    screen, path = setup
    trainer = ReferenceTrainer(load_config(path))
    trainer.update()
    d = copy.deepcopy(trainer.graph.models['discriminator'].state_dict())
    prior = copy.deepcopy(trainer.prior.state_dict())
    d_opt = copy.deepcopy(trainer.opt_d.state_dict())
    p_opt = [copy.deepcopy(trainer.opt_g.state[p]) for p in trainer.program.prior_parameters]
    g = [p.detach().clone() for p in trainer.program.generator_parameters]
    penalty_rng = trainer.streams['penalty'].get_state().clone()
    prior_ema = copy.deepcopy(trainer.ema_prior.state_dict())
    screen.extra_generator_update(trainer)
    assert trainer.step == 1
    assert _same_state(d, trainer.graph.models['discriminator'].state_dict())
    assert _same_state(prior, trainer.prior.state_dict())
    assert _same_state(d_opt, trainer.opt_d.state_dict())
    assert _same_state(p_opt, [trainer.opt_g.state[p] for p in trainer.program.prior_parameters])
    assert _same_state(prior_ema, trainer.ema_prior.state_dict())
    assert torch.equal(penalty_rng, trainer.streams['penalty'].get_state())
    assert any(not torch.equal(p, old) for p, old in zip(trainer.program.generator_parameters, g))


def test_schedule_counts_transition_and_full_restoration(setup):
    screen, path = setup
    report = screen.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=4,
                              prepare=screen.prepare(ratio=4, warmup_rounds=2, warmup_g_factor=.25))
    assert report['status'] == 'complete', report.get('failure')
    assert report['restored'] and report['source_config_unchanged']
    proposal = report['proposal']
    assert (proposal['g_updates'], proposal['d_updates'], proposal['prior_updates']) == (10, 4, 4)
    assert [len(r['extra_g']) for r in proposal['rounds']] == [3, 3, 0, 0]
    assert [r['g_lr'] for r in proposal['rounds']] == [2.5e-5, 2.5e-5, 1e-4, 1e-4]


def test_ratio_one_is_exact_native_trajectory(setup):
    screen, path = setup
    native = screen.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=3)
    wrapped = screen.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=3,
                               prepare=screen.prepare(ratio=1, warmup_rounds=2))
    assert native['rollout_final_parameters_sha256'] == wrapped['rollout_final_parameters_sha256']
    assert native['per_step'] == wrapped['per_step']


def test_penalty_only_step_has_its_own_moments_and_cannot_update_g_or_prior(setup):
    screen, path = setup
    trainer = ReferenceTrainer(load_config(path))
    trainer.update()  # Populate adversarial Adam moments before testing isolation.
    d_opt = copy.deepcopy(trainer.opt_d.state_dict())
    g = [p.detach().clone() for p in trainer.program.generator_parameters]
    prior = copy.deepcopy(trainer.prior.state_dict())
    d = [p.detach().clone() for p in trainer.program.critic_parameters]
    optimizer = screen.DeviceAdam(trainer.program.critic_parameters, **trainer.opt_d.defaults)
    penalty = screen.GradientPenalty(arm='e_interp', coeff=1, lazy_k=1)
    with torch.no_grad():
        batch, _, context = trainer._draw(None, None)
    result = screen.penalty_only_update(trainer, optimizer, penalty, batch, context)
    assert result['optimizer_step'] and result['loss'] > 0
    assert _same_state(d_opt, trainer.opt_d.state_dict())
    assert _same_state(prior, trainer.prior.state_dict())
    assert all(torch.equal(p, old) for p, old in zip(trainer.program.generator_parameters, g))
    assert any(not torch.equal(p, old) for p, old in zip(trainer.program.critic_parameters, d))
    # A zero penalty must not move D even after penalty momentum exists.
    d = [p.detach().clone() for p in trainer.program.critic_parameters]
    old_state = copy.deepcopy(optimizer.state_dict())
    result = screen.penalty_only_update(trainer, optimizer,
                                        screen.GradientPenalty(arm='b_cap', kappa=1e9), batch, context)
    assert result == {'loss': 0., 'optimizer_step': False}
    assert _same_state(old_state, optimizer.state_dict())
    assert all(torch.equal(p, old) for p, old in zip(trainer.program.critic_parameters, d))


def test_interpolation_warmup_restores_and_stops_penalty_at_transition(setup):
    screen, path = setup
    report = screen.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=3,
                              prepare=screen.prepare(ratio=2, warmup_rounds=2, extra_penalty='e_interp'))
    assert report['status'] == 'complete', report.get('failure')
    assert report['restored']
    proposal = report['proposal']
    assert proposal['penalty_only_updates'] == 2
    assert proposal['g_updates'] == 5 and proposal['d_updates'] == 3
    assert proposal['rounds'][-1]['extra_g'] == []
