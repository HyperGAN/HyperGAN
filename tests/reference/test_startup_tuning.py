"""Exact startup persistence, EMA and checkpoint recovery of calibrated tensors."""
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
            progress({'trial_step': 0, 'trial_steps': 8, 'lr_factor': .3})
        return {'outcome': 'selected', 'selected_g_lr_factor': .3,
                'selected_candidate': 'fixture_g_lr', 'reason': 'fixture confirmation passed',
                'disposable_completed_updates': 8}
    monkeypatch.setattr(module, 'tune_startup_dynamics', selected)
    return module


def _candidate(trainer, *, progress=None):
    owned = {id(p) for p in trainer.program.generator_parameters}
    name, parameter = next((n, p) for n, p in trainer.graph.named_parameters() if id(p) in owned)
    with torch.no_grad():
        parameter.mul_(.75)
    if progress:
        progress({'candidate': 1, 'total_candidates': 1})
    return {'outcome': 'selected', 'selected_candidate': 'fixture',
            'transformations': [{'path': name, 'factor': .75}]}


def test_tuned_initial_checkpoint_contains_exact_parameters_and_resume_does_not_retune(tmp_path, monkeypatch):
    import hypergan.initialization_tuning as numerical
    monkeypatch.setattr(numerical, 'tune_initialization', _candidate)
    config = write_default(tmp_path / 'project', device='cpu')
    original = config.read_bytes()
    root = tmp_path / 'run'
    result = train(config, root, steps=2, tune=True, stop_after_steps=1)
    provenance = result['initialization_tuning']
    report = json.loads((root / 'tuning/report.json').read_text())
    assert len(report['ema_synchronized_parameters']) == 1
    assert report['initialization_optimizer_steps'] == report['retained_training_updates'] == 0
    assert report['disposable_trial_updates'] == 8 and 'optimizer_steps' not in report
    assert hashlib.sha256((root / 'tuning/report.json').read_bytes()).hexdigest() == provenance['report_sha256']
    generations = sorted((root / 'checkpoints').glob('*-step-*'))
    initial = next(p for p in generations if json.loads((p / 'manifest.json').read_text())['step'] == 0)
    _, metadata, state = read_checkpoint(root, initial)
    assert metadata['initialization_tuning'] == provenance
    assert state['base_lrs'][0][0] == load_config(config)['optimizer']['lr'] * .3
    assert provenance['selected_g_lr_factor'] == .3
    assert not state['optimizers'][0]['state'] and not state['optimizers'][1]['state']
    name = report['ema_synchronized_parameters'][0]
    assert torch.equal(state['graph'][name], state['ema_graph'][name])
    baseline = ReferenceTrainer(load_config(config))
    torch.testing.assert_close(state['graph'][name], baseline.graph.state_dict()[name] * .75, rtol=0, atol=0)
    monkeypatch.setattr(numerical, 'tune_initialization', lambda *a, **k: pytest.fail('resume retuned'))
    with pytest.warns(RuntimeWarning, match='tuning is skipped'):
        result = train(config, root, steps=2, tune=True)
    assert result['status'] == 'complete'
    assert config.read_bytes() == original
    final_state = read_checkpoint(root)[2]
    assert final_state['base_lrs'][0][0] == baseline.base_lrs[0][0] * .3
    assert final_state['base_lrs'][0][1:] == baseline.base_lrs[0][1:]
    assert final_state['base_lrs'][1] == baseline.base_lrs[1]


def test_artifact_failure_restores_owned_parameters_and_ema(tmp_path, monkeypatch):
    import hypergan.initialization_tuning as numerical
    import hypergan.startup_tuning as persistence
    config = write_default(tmp_path / 'project', device='cpu')
    trainer = ReferenceTrainer(load_config(config))
    root = tmp_path / 'run'
    root.mkdir()
    rates = deepcopy(trainer.base_lrs)
    original = {name: p.detach().clone() for name, p in trainer.graph.named_parameters()}
    ema = {name: p.detach().clone() for name, p in trainer.ema_graph.named_parameters()}
    monkeypatch.setattr(numerical, 'tune_initialization', _candidate)
    atomic = persistence.atomic_json
    def failing(path, value):
        if path.name == 'report.json':
            raise OSError('report storage failed')
        return atomic(path, value)
    monkeypatch.setattr(persistence, 'atomic_json', failing)
    with pytest.raises(OSError, match='report storage failed'):
        persistence.tune_initialized(trainer, root)
    assert trainer.base_lrs == rates
    assert [[group['lr'] for group in optimizer.param_groups] for optimizer in (trainer.opt_g, trainer.opt_d)] == rates
    for name, parameter in trainer.graph.named_parameters():
        assert torch.equal(parameter, original[name])
    for name, parameter in trainer.ema_graph.named_parameters():
        assert torch.equal(parameter, ema[name])


def test_tuning_provenance_supports_scalar_owned_parameters(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    import hypergan.initialization_tuning as numerical
    from hypergan.startup_tuning import tune_initialized
    config = load_config(write_default(tmp_path / 'project', device='cpu'))
    graph = torch.nn.Module()
    graph.register_parameter('gain', torch.nn.Parameter(torch.tensor(1.0)))
    rate = config['optimizer']['lr']
    d_rate = rate * config['optimizer']['d_lr_mult']
    trainer = SimpleNamespace(step=0, opt_g=SimpleNamespace(state={}, param_groups=[{'lr': rate}]),
                              opt_d=SimpleNamespace(state={}, param_groups=[{'lr': d_rate}]),
                              base_lrs=[[rate], [d_rate]], config=config, graph=graph, ema_graph=deepcopy(graph),
                              program=SimpleNamespace(generator_parameters=tuple(graph.parameters())))
    monkeypatch.setattr(numerical, 'tune_initialization', lambda *a, **k: {
        'outcome': 'kept_baseline', 'selected_candidate': 'baseline', 'transformations': []})
    root = tmp_path / 'run'
    root.mkdir()
    result = tune_initialized(trainer, root)
    assert len(result['selected_parameters_sha256']) == 64
    assert trainer.graph.gain.item() == trainer.ema_graph.gain.item() == 1.0


@pytest.mark.parametrize('outcome', ['kept_baseline', 'unresolved', 'skipped'])
def test_unselected_dynamics_decision_keeps_all_absolute_rates(tmp_path, monkeypatch, dynamics, outcome):
    import hypergan.initialization_tuning as numerical
    from hypergan.startup_tuning import tune_initialized
    monkeypatch.setattr(numerical, 'tune_initialization', _candidate)
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: {
        'outcome': outcome, 'selected_g_lr_factor': 1., 'reason': 'fixture decision'})
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'project', device='cpu')))
    rates = deepcopy(trainer.base_lrs)
    root = tmp_path / 'run'
    root.mkdir()
    result = tune_initialized(trainer, root)
    assert trainer.base_lrs == rates
    assert [[group['lr'] for group in optimizer.param_groups] for optimizer in (trainer.opt_g, trainer.opt_d)] == rates
    assert result['dynamics_outcome'] == outcome and result['selected_g_lr_factor'] == 1.
    assert result['dynamics_reason'] == 'fixture decision'


def test_dynamics_runs_after_initial_ema_sync_and_is_visible_in_progress(tmp_path, monkeypatch, dynamics):
    import hypergan.initialization_tuning as numerical
    from hypergan.startup_tuning import tune_initialized
    monkeypatch.setattr(numerical, 'tune_initialization', _candidate)
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'project', device='cpu')))
    def select(trainer, *, progress=None):
        assert trainer.step == 0 and not trainer.opt_g.state and not trainer.opt_d.state
        for name, parameter in trainer.graph.named_parameters():
            assert torch.equal(parameter, dict(trainer.ema_graph.named_parameters())[name])
        progress({'trial_step': 8, 'trial_steps': 8})
        return {'outcome': 'selected', 'selected_g_lr_factor': .271828, 'reason': 'confirmed'}
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', select)
    root = tmp_path / 'run'
    root.mkdir()
    events = []
    result = tune_initialized(trainer, root, on_event=events.append)
    assert [event['phase'] for event in events] == ['initialization', 'dynamics']
    assert result['selected_g_lr_factor'] == .271828
    overrides = json.loads((root / 'tuning/overrides.json').read_text())
    assert overrides['optimizer_override'] == result['optimizer_override']


def test_tuned_rate_split_resume_matches_uninterrupted_training(tmp_path, monkeypatch, dynamics):
    import hypergan.initialization_tuning as numerical
    from .test_recovery import equal
    monkeypatch.setattr(numerical, 'tune_initialization', _candidate)
    config = write_default(tmp_path / 'project', device='cpu')
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    train(config, whole, steps=4, tune=True)
    train(config, split, steps=4, tune=True, stop_after_steps=2)
    monkeypatch.setattr(numerical, 'tune_initialization', lambda *a, **k: pytest.fail('resume reinitialized'))
    monkeypatch.setattr(dynamics, 'tune_startup_dynamics', lambda *a, **k: pytest.fail('resume reran dynamics'))
    with pytest.warns(RuntimeWarning, match='tuning is skipped'):
        train(config, split, steps=4, tune=True)
    equal(read_checkpoint(whole)[2], read_checkpoint(split)[2])
