"""Run-local measured rates, unchanged initialization, and exact recovery."""
import hashlib
from copy import deepcopy
import json

import pytest
import torch

from hypergan.checkpoints import read_checkpoint
from hypergan.config import load_config, write_default
from hypergan.execution import train
from hypergan.training import ReferenceTrainer


@pytest.fixture(autouse=True)
def dynamics(monkeypatch):
    import hypergan.startup_dynamics as module
    def selected(trainer, *, progress=None):
        if progress:
            progress({'stage': 'validate', 'trial_steps': 8, 'g_lr_factor': .3, 'd_lr_factor': .5})
        return {'outcome': 'selected', 'selected_g_lr_factor': .3, 'selected_d_lr_factor': .5,
                'reason': 'fixture held-out and replay checks passed', 'disposable_completed_updates': 16}
    monkeypatch.setattr(module, 'tune_startup_dynamics', selected)
    return module


def test_tuned_initial_checkpoint_preserves_weights_and_only_overrides_rates(tmp_path, monkeypatch, dynamics):
    from .test_recovery import equal
    config = write_default(tmp_path / 'project', device='cpu')
    original = config.read_bytes()
    baseline = ReferenceTrainer(load_config(config))
    root = tmp_path / 'run'
    result = train(config, root, steps=2, tune=True, stop_after_steps=1)
    provenance = result['initialization_tuning']
    assert provenance['method'] == 'measured-update-response'
    assert 'g_lr_warmup' not in provenance
    report = json.loads((root / 'tuning/report.json').read_text())
    assert report['retained_training_updates'] == 0
    assert report['disposable_trial_updates'] == 16
    assert 'transformations' not in report and 'ema_synchronized_parameters' not in report
    assert hashlib.sha256((root / 'tuning/report.json').read_bytes()).hexdigest() == provenance['report_sha256']
    initial = next(p for p in (root / 'checkpoints').glob('*-step-*')
                   if json.loads((p / 'manifest.json').read_text())['step'] == 0)
    _, metadata, state = read_checkpoint(root, initial)
    assert metadata['initialization_tuning'] == provenance
    equal(state['graph'], baseline.graph.state_dict())
    equal(state['ema_graph'], baseline.ema_graph.state_dict())
    equal(state['prior'], baseline.prior.state_dict())
    assert state['base_lrs'][0][0] == baseline.base_lrs[0][0] * .3
    assert state['base_lrs'][1][0] == baseline.base_lrs[1][0] * .5
    assert state['base_lrs'][0][1:] == baseline.base_lrs[0][1:]
    assert not state['optimizers'][0]['state'] and not state['optimizers'][1]['state']
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: pytest.fail('resume retuned'))
    with pytest.warns(RuntimeWarning, match='tuning is skipped'):
        resumed = train(config, root, steps=2, tune=True)
    assert resumed['status'] == 'complete'
    assert resumed['initialization_tuning'] == provenance
    assert config.read_bytes() == original


@pytest.mark.parametrize('factors', [(.3, 1.), (1., .5), (.3, .5)])
def test_tuned_rate_split_resume_matches_uninterrupted_training(tmp_path, monkeypatch, dynamics, factors):
    from .test_recovery import equal
    g, d = factors
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: {
        'outcome': 'selected', 'selected_g_lr_factor': g, 'selected_d_lr_factor': d,
        'reason': 'fixture confirmation passed', 'disposable_completed_updates': 16})
    config = write_default(tmp_path / 'project', device='cpu')
    original = config.read_bytes()
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    train(config, whole, steps=6, tune=True)
    result = train(config, split, steps=6, tune=True, stop_after_steps=2)
    tuning = result['initialization_tuning']
    assert tuning['selected_g_lr_factor'] == g and tuning['selected_d_lr_factor'] == d
    assert 'g_lr_warmup' not in tuning
    report = json.loads((split / 'tuning/report.json').read_text())
    override = json.loads((split / 'tuning/overrides.json').read_text())
    assert report['optimizer_override'] == override['optimizer_override'] == tuning['optimizer_override']
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: pytest.fail('resume retuned'))
    resumed = train(config, split, steps=6)
    assert resumed['initialization_tuning'] == tuning
    equal(read_checkpoint(whole)[2], read_checkpoint(split)[2])
    assert config.read_bytes() == original


@pytest.mark.parametrize('outcome', ['kept_baseline', 'unresolved', 'skipped'])
def test_unselected_decision_keeps_all_absolute_rates(tmp_path, monkeypatch, dynamics, outcome):
    from hypergan.startup_tuning import tune_initialized
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: {
        'outcome': outcome, 'selected_g_lr_factor': 1., 'selected_d_lr_factor': 1.,
        'reason': 'fixture decision'})
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'project', device='cpu')))
    rates = deepcopy(trainer.base_lrs)
    root = tmp_path / 'run'
    root.mkdir()
    result = tune_initialized(trainer, root)
    assert trainer.base_lrs == rates
    assert [[group['lr'] for group in optimizer.param_groups] for optimizer in (trainer.opt_g, trainer.opt_d)] == rates
    assert result['dynamics_outcome'] == outcome and result['selected_g_lr_factor'] == 1.
    assert result['dynamics_reason'] == 'fixture decision'


def test_progress_is_one_measured_update_pipeline(tmp_path, monkeypatch, dynamics):
    from hypergan.startup_tuning import tune_initialized
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'project', device='cpu')))
    def select(trainer, *, progress=None):
        assert trainer.step == 0 and not trainer.opt_g.state and not trainer.opt_d.state
        assert getattr(trainer, 'g_lr_warmup', None) is None
        progress({'stage': 'fit', 'trial_step': 8, 'trial_steps': 8})
        return {'outcome': 'selected', 'selected_g_lr_factor': .271828, 'reason': 'confirmed'}
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', select)
    root = tmp_path / 'run'
    root.mkdir()
    events = []
    result = tune_initialized(trainer, root, on_event=events.append)
    assert [event['phase'] for event in events] == ['dynamics']
    assert events[0]['stage'] == 'fit' and events[0]['method'] == 'measured-update-response'
    assert result['selected_g_lr_factor'] == .271828
    assert getattr(trainer, 'g_lr_warmup', None) is None


def test_selected_rates_stay_selected_across_former_warmup_endpoint(tmp_path, dynamics):
    """Exercise scheduling at 999/1000/1001 without a thousand training updates."""
    from hypergan.startup_tuning import tune_initialized
    config = load_config(write_default(tmp_path / 'project', device='cpu'))
    config['training'].update(steps=2000, lr_floor=1.)
    trainer = ReferenceTrainer(config)
    original_rates = deepcopy(trainer.base_lrs)
    root = tmp_path / 'run'
    root.mkdir()
    result = tune_initialized(trainer, root)
    expected_rates = deepcopy(original_rates)
    expected_rates[0][0] *= .3
    expected_rates[1][0] *= .5
    trainer.step = 998
    for expected_step in (999, 1000, 1001):
        trainer.update()
        assert trainer.step == expected_step
        actual_rates = [[group['lr'] for group in optimizer.param_groups]
                        for optimizer in (trainer.opt_g, trainer.opt_d)]
        assert actual_rates == expected_rates
        assert trainer.base_lrs == expected_rates
    assert 'g_lr_warmup' not in result
    assert getattr(trainer, 'g_lr_warmup', None) is None


@pytest.mark.parametrize('g_factor,d_factor', [(.3, 1.), (1., .5), (.3, .5)])
def test_artifact_failure_restores_all_rates_without_changing_weights(tmp_path, monkeypatch, dynamics, g_factor, d_factor):
    from .test_recovery import equal
    import hypergan.startup_tuning as persistence
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'project', device='cpu')))
    rates = deepcopy(trainer.base_lrs)
    graph = deepcopy(trainer.graph.state_dict())
    ema = deepcopy(trainer.ema_graph.state_dict())
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: {
        'outcome': 'selected', 'selected_g_lr_factor': g_factor, 'selected_d_lr_factor': d_factor})
    atomic = persistence.atomic_json
    def fail(path, value):
        if path.name == 'report.json':
            assert trainer.base_lrs[0][0] == rates[0][0] * g_factor
            assert trainer.base_lrs[1][0] == rates[1][0] * d_factor
            raise OSError('report storage failed')
        return atomic(path, value)
    monkeypatch.setattr(persistence, 'atomic_json', fail)
    root = tmp_path / 'run'
    root.mkdir()
    with pytest.raises(OSError, match='report storage failed'):
        persistence.tune_initialized(trainer, root)
    assert trainer.base_lrs == rates
    assert [[g['lr'] for g in opt.param_groups] for opt in (trainer.opt_g, trainer.opt_d)] == rates
    equal(trainer.graph.state_dict(), graph)
    equal(trainer.ema_graph.state_dict(), ema)


def test_inconsistent_selected_outcome_never_installs_override(tmp_path, monkeypatch, dynamics):
    from hypergan.startup_tuning import tune_initialized
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'project', device='cpu')))
    rates = deepcopy(trainer.base_lrs)
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: {
        'outcome': 'unresolved', 'selected_g_lr_factor': .3, 'selected_d_lr_factor': .5})
    root = tmp_path / 'run'
    root.mkdir()
    with pytest.raises(ValueError, match='selection outcome'):
        tune_initialized(trainer, root)
    assert trainer.base_lrs == rates
