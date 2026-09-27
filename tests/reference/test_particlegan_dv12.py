"""ParticleGAN#217 builds: the DV12 LR controller and KA2 penalty as GANTrainer trains them."""
import copy

import pytest
import torch

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import resolve_config
from hypergan.training import LR_CONTROLLER, ReferenceTrainer, particlegan_recipe

dv12 = pytest.mark.skipif(not LR_CONTROLLER, reason='needs a DV12 ParticleGAN build (ParticleGAN#217)')
NET = 'linear(16)\nleaky_relu(0.2)\nlinear()\n'


def _config(**training):
    return resolve_config({
        'defaults': 'particlegan', 'name': 'test/dv12',
        'components': {'generator': {'factory': 'hndl', 'inputs': {'x': 'latent'},
                                     'args': {'source': NET, 'input_shape': ['B', 2], 'output_shape': ['B', 2]}},
                       'discriminator': {'factory': 'hndl', 'inputs': {'x': 'candidate'},
                                         'args': {'source': NET, 'input_shape': ['B', 2], 'output_shape': ['B', 1]}}},
        'prior': {'kind': 'particles', 'args': {'num_particles': 64, 'z_dim': 2}},
        'training': {'steps': 40, 'batch_size': 8, 'seed': 0, 'phase_draws': 'independent', **training}})


def _reals(n):
    rng = torch.Generator().manual_seed(1)
    return [torch.randn(8, 2, generator=rng) * 2 for _ in range(n)]


@pytest.fixture
def short_warmup(monkeypatch):
    """Reach KA2's blended phase after 5 applied calls instead of 799."""
    from particlegan import grad_regularizers
    monkeypatch.setattr(grad_regularizers, 'WARMUP_CALLS', 6)


def _step(trainer, real):
    return trainer.update({'real': real}, generator_batch={'real': real})[0]


@dv12
def test_reference_trainer_is_gan_trainer_bit_for_bit(short_warmup):
    """Same weights and streams: every loss, applied LR and controller value, and the final state."""
    from particlegan import GANTrainer
    trainer = ReferenceTrainer(_config())
    controller = trainer.lr_controller
    assert trainer.opt_g.controller is controller and trainer.gan.controller is controller
    assert trainer.penalty.regularizer.controller is controller and trainer.prior.support_jitter
    recipe = trainer.recipe.replace(num_particles=64, z_dim=2, batch_size=8)
    reference = GANTrainer(recipe, copy.deepcopy(trainer.graph.models['generator']),
                           copy.deepcopy(trainer.graph.models['discriminator']),
                           prior=copy.deepcopy(trainer.prior), seed=0)
    for real in _reals(20):
        row, out = _step(trainer, real), reference.step(real, collect_stats=True)
        assert (row['d_loss'], row['g_loss'], row['gradient_penalty']) == tuple(
            float(out[key]) for key in ('loss_d', 'loss_g', 'penalty'))
        assert trainer.opt_g.applied_lrs == reference.opt_g.applied_lrs
        assert trainer.opt_d.applied_lrs == reference.opt_d.applied_lrs
        assert [row['lr_generator'], row['lr_prior'], row['lr_critic']] == [*reference.opt_g.applied_lrs, *reference.opt_d.applied_lrs]
        assert row['lr_scale'] == 1.0
        assert controller.diagnostics() == reference.opt_d.controller.diagnostics()
    assert out['penalty_stats']['phase'] == 'blend' and trainer.opt_d.record.state_dict() == reference.opt_d.record.state_dict()
    pairs = [(trainer.graph.models['generator'], reference.G), (trainer.graph.models['discriminator'], reference.D),
             (trainer.prior, reference.prior), (trainer.opt_d.ema_critic['discriminator'], reference.ema_D)]
    for mine, theirs in pairs:
        for a, b in zip(mine.state_dict().values(), theirs.state_dict().values()):
            assert torch.equal(a, b)
    assert torch.equal(trainer.opt_g.latent_history, reference.opt_g.latent_history)


@dv12
def test_resume_inside_the_blend_is_exact(short_warmup):
    reals = _reals(16)
    full = ReferenceTrainer(_config())
    for real in reals:
        _step(full, real)
    first = ReferenceTrainer(_config())
    for real in reals[:10]:
        _step(first, real)
    assert first.opt_d.record.anchor_started
    state = copy.deepcopy(trainer_state(first, {'real': reals[9]}))
    resumed = ReferenceTrainer(_config())
    restore_trainer(resumed, state)
    for real in reals[10:]:
        _step(resumed, real)
    assert resumed.lr_controller.state_dict().keys() == full.lr_controller.state_dict().keys()
    assert resumed.lr_controller.diagnostics() == full.lr_controller.diagnostics()
    for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
        for key, value in getattr(resumed, name).state_dict().items():
            assert torch.equal(value, getattr(full, name).state_dict()[key]), (name, key)


def test_a_formulation_change_is_named_before_other_resume_checks():
    trainer = ReferenceTrainer(_config(steps=4))
    real = _reals(1)[0]
    _step(trainer, real)
    state = copy.deepcopy(trainer_state(trainer, {'real': real}))
    regularizer = state['optimizers'][1]['regularizer']
    if 'controller' in regularizer:
        del regularizer['controller']
    else:
        regularizer['controller'] = {}
    with pytest.raises(ValueError, match='ParticleGAN build that wrote it'):
        restore_trainer(ReferenceTrainer(_config(steps=4)), state)


def test_penalty_fields_the_build_lacks_are_refused():
    config = _config()
    if LR_CONTROLLER:
        assert config['gradient_penalty']['anchor_min_decay'] == particlegan_recipe(config).reg_anchor_min_decay
        config['gradient_penalty']['anchor_decay'] = 0.99
        match = 'anchor_decay must be omitted'
    else:
        assert 'anchor_min_decay' not in config['gradient_penalty']
        config['gradient_penalty']['anchor_min_decay'] = 0.9
        match = 'anchor_min_decay must be omitted'
    with pytest.raises(ValueError, match=match):
        particlegan_recipe(config)


def test_train_rows_record_the_applied_learning_rates():
    trainer = ReferenceTrainer(_config())
    row = _step(trainer, _reals(1)[0])
    for optimizer, keys in ((trainer.opt_g, ('lr_generator', 'lr_prior')), (trainer.opt_d, ('lr_critic',))):
        applied = optimizer.applied_lrs if LR_CONTROLLER else [group['lr'] for group in optimizer.param_groups]
        assert [row[key] for key in keys] == applied
