"""Deterministic startup calibration safety and numerical behavior (no seed sweep)."""
import copy

import pytest
import torch
from torch import nn

from hypergan.checkpoints import capture_rng
from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer
from hypergan.initialization_tuning import _inventory, _acceptable, tune_initialization


def trainer():
    return ReferenceTrainer(resolve_config({}))


def snapshot(t):
    return {name: p.detach().clone() for name, p in t.graph.named_parameters()}


def test_calibration_selects_guarded_improvement_and_preserves_other_state():
    t = trainer()
    before = snapshot(t)
    prior = copy.deepcopy(t.prior.state_dict())
    streams = {k: v.get_state().clone() for k, v in t.streams.items()}
    rng = capture_rng()
    # Existing gradients are not a target or an input to initialization tuning.
    for p in t.graph.parameters():
        p.grad = torch.ones_like(p)
    report = tune_initialization(t)
    assert report['outcome'] == 'selected'
    assert report['heldout_validation']['accepted']
    assert report['after']['transmission']['score'] < report['before']['transmission']['score']
    transforms = {row['path']: row['factor'] for row in report['transformations']}
    for name, p in t.graph.named_parameters():
        expected = before[name] * transforms.get(name, 1.)
        torch.testing.assert_close(p, expected, rtol=0, atol=0)
        assert torch.equal(p.grad, torch.ones_like(p))
    for name, value in t.prior.state_dict().items():
        assert torch.equal(value, prior[name])
    for name, stream in t.streams.items():
        assert torch.equal(stream.get_state(), streams[name])
    after_rng = capture_rng()
    assert torch.equal(rng['torch'], after_rng['torch'])
    assert rng['python'] == after_rng['python'] and rng['numpy'] == after_rng['numpy']
    assert not t.opt_g.state and not t.opt_d.state and t.step == 0
    assert report['protected_state_verification']['unchanged']


def test_rejected_candidates_restore_exact_baseline(monkeypatch):
    import hypergan.initialization_tuning as tuning
    t = trainer()
    before = snapshot(t)
    monkeypatch.setattr(tuning, '_acceptable', lambda *args: ['test_rejection'])
    report = tune_initialization(t)
    assert report['outcome'] == 'kept_baseline' and report['transformations'] == []
    assert all(torch.equal(before[n], p) for n, p in t.graph.named_parameters())


def test_exception_after_candidate_edit_rolls_back_every_tensor(monkeypatch):
    import hypergan.initialization_tuning as tuning
    t = trainer()
    before = snapshot(t)
    rng = capture_rng()
    original = tuning._probe
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('candidate probe failed')
        return original(*args, **kwargs)
    monkeypatch.setattr(tuning, '_probe', fail)
    with pytest.raises(RuntimeError, match='candidate probe failed'):
        tune_initialization(t)
    assert calls == 2
    assert all(torch.equal(before[n], p) for n, p in t.graph.named_parameters())
    assert torch.equal(capture_rng()['torch'], rng['torch'])


def test_pretrained_trainable_descendants_and_storage_aliases_excluded():
    from hndl.operators.pretrained import Pretrained
    t = trainer()
    g = t.graph.models['generator']
    # Construction-free native marker, avoiding downloads/provider loading.
    frozen = Pretrained.__new__(Pretrained)
    nn.Module.__init__(frozen)
    frozen.inner = nn.Linear(4, 4)
    g.add_module('pretrained_fixture', frozen)
    from types import SimpleNamespace
    t.program = SimpleNamespace(generator_parameters=tuple(t.program.generator_parameters) + tuple(frozen.parameters()))
    layers, _ = _inventory(t)
    assert all('pretrained_fixture' not in name for name, _ in layers)
    first_name, first = layers[0]
    # A different Parameter object sharing pretrained storage is protected too.
    frozen.register_parameter('tied', nn.Parameter(first.weight.detach()))
    layers, _ = _inventory(t)
    assert first_name not in dict(layers)


def test_uncertain_custom_generator_ownership_does_not_calibrate():
    t = trainer()
    t.config['components']['generator']['factory'] = 'custom:Generator'
    before = snapshot(t)
    report = tune_initialization(t)
    assert report['outcome'] == 'kept_baseline'
    assert report['ownership_exclusion']
    assert all(torch.equal(before[n], p) for n, p in t.graph.named_parameters())


def test_stateful_data_cursor_is_restored():
    t = trainer()
    source = t.data
    class Stateful:
        calls = 0
        def __call__(self, *args, **kwargs):
            self.calls += 1
            return source(*args, **kwargs)
        def state_dict(self):
            return {'calls': self.calls}
        def load_state_dict(self, state):
            self.calls = state['calls']
    t.data = Stateful()
    tune_initialization(t)
    assert t.data.calls == 0


def test_no_tuning_after_optimizer_update():
    t = trainer()
    t.update()
    with pytest.raises(ValueError, match='before the first optimizer'):
        tune_initialization(t)


def test_better_transmission_cannot_buy_output_collapse():
    baseline = dict(score=1., output_rms=1., output_std=1., sample_diversity_rms=1., nonfinite_output_fraction=0.)
    candidate = dict(baseline, score=.1, sample_diversity_rms=.01)
    diagnostic = {'loss': 1., 'summary': {'nonfinite_activation_records': 0, 'generated_output_gradient_rms': 1.}}
    assert 'sample_diversity_rms_outside_guard' in _acceptable(baseline, candidate, diagnostic)


def test_parameter_overflow_and_disconnected_generator_are_rejected():
    baseline = dict(score=1., output_rms=1., output_std=1., sample_diversity_rms=1., nonfinite_output_fraction=0.)
    candidate = dict(baseline, score=.1)
    diagnostic = {'loss': 1., 'summary': {'nonfinite_activation_records': 0, 'generated_output_gradient_rms': 1.},
                  'parameters': [{'path': 'models.generator.weight', 'owner': 'generator_or_auxiliary', 'status': 'nonfinite'}]}
    reasons = _acceptable(baseline, candidate, diagnostic)
    assert 'nonfinite_parameter_gradient' in reasons
    assert 'missing_generator_parameter_signal' in reasons


def test_pretrained_buffer_mutation_is_detected_before_restoring(monkeypatch):
    from hndl.operators.pretrained import Pretrained
    import hypergan.initialization_tuning as tuning
    t = trainer()
    frozen = Pretrained.__new__(Pretrained)
    nn.Module.__init__(frozen)
    frozen.register_buffer('running', torch.ones(1))
    t.graph.models['generator'].add_module('pretrained_fixture', frozen)
    before = snapshot(t)
    original = tuning._structural
    def mutate(*args, **kwargs):
        frozen.running.add_(1.)
        return original(*args, **kwargs)
    monkeypatch.setattr(tuning, '_structural', mutate)
    with pytest.raises(ValueError, match='changed pretrained state'):
        tune_initialization(t)
    assert torch.equal(frozen.running, torch.ones(1))
    assert all(torch.equal(before[n], p) for n, p in t.graph.named_parameters())


def test_uncertain_custom_subtree_is_not_calibratable():
    from types import SimpleNamespace
    t = trainer()
    class UnknownOwner(nn.Module):
        def __init__(self):
            super().__init__()
            self.affine = nn.Linear(4, 4)
    unknown = UnknownOwner()
    t.graph.models['generator'].add_module('unknown', unknown)
    t.program = SimpleNamespace(generator_parameters=tuple(t.program.generator_parameters) + tuple(unknown.parameters()))
    layers, _ = _inventory(t)
    assert all('unknown' not in name for name, _ in layers)


def test_saturated_tanh_boundary_is_improved_and_confirmed_on_heldout_draw():
    config = resolve_config({})
    config['components']['generator']['args']['source'] = 'linear(64)\nleaky_relu(0.2)\nlinear(2)\ntanh()'
    t = ReferenceTrainer(config)
    final = [m for m in t.graph.models['generator'].modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        final.weight.mul_(20.)
    report = tune_initialization(t)
    assert report['outcome'] == 'selected'
    assert report['selected_candidate'] == 'output_boundary'
    assert report['heldout_validation']['accepted']
    before, after = (report[key]['transmission'] for key in ('before', 'after'))
    assert after['score'] < .75 * before['score']
    assert after['absolute_output_above_0_99_fraction'] < before['absolute_output_above_0_99_fraction']
    assert after['sample_diversity_rms'] >= .5 * before['sample_diversity_rms']
