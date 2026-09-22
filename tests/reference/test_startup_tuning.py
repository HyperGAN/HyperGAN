"""Exact startup persistence, EMA and checkpoint recovery of calibrated tensors."""
import hashlib
import json

import pytest
import torch

from hypergan.checkpoints import read_checkpoint
from hypergan.config import load_config, write_default
from hypergan.execution import train
from hypergan.training import ReferenceTrainer


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
    assert hashlib.sha256((root / 'tuning/report.json').read_bytes()).hexdigest() == provenance['report_sha256']
    generations = sorted((root / 'checkpoints').glob('*-step-*'))
    initial = next(p for p in generations if json.loads((p / 'manifest.json').read_text())['step'] == 0)
    _, metadata, state = read_checkpoint(root, initial)
    assert metadata['initialization_tuning'] == provenance
    name = report['ema_synchronized_parameters'][0]
    assert torch.equal(state['graph'][name], state['ema_graph'][name])
    baseline = ReferenceTrainer(load_config(config))
    torch.testing.assert_close(state['graph'][name], baseline.graph.state_dict()[name] * .75, rtol=0, atol=0)
    monkeypatch.setattr(numerical, 'tune_initialization', lambda *a, **k: pytest.fail('resume retuned'))
    with pytest.warns(RuntimeWarning, match='tuning is skipped'):
        result = train(config, root, steps=2, tune=True)
    assert result['status'] == 'complete'
    assert config.read_bytes() == original


def test_artifact_failure_restores_owned_parameters_and_ema(tmp_path, monkeypatch):
    import hypergan.initialization_tuning as numerical
    import hypergan.startup_tuning as persistence
    config = write_default(tmp_path / 'project', device='cpu')
    trainer = ReferenceTrainer(load_config(config))
    root = tmp_path / 'run'
    root.mkdir()
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
    trainer = SimpleNamespace(step=0, opt_g=SimpleNamespace(state={}), opt_d=SimpleNamespace(state={}),
                              config=config, graph=graph, ema_graph=deepcopy(graph),
                              program=SimpleNamespace(generator_parameters=tuple(graph.parameters())))
    monkeypatch.setattr(numerical, 'tune_initialization', lambda *a, **k: {
        'outcome': 'kept_baseline', 'selected_candidate': 'baseline', 'transformations': []})
    root = tmp_path / 'run'
    root.mkdir()
    result = tune_initialized(trainer, root)
    assert len(result['selected_parameters_sha256']) == 64
    assert trainer.graph.gain.item() == trainer.ema_graph.gain.item() == 1.0
