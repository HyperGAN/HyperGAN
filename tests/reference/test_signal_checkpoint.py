"""Checkpoint diagnostics load online G/D without changing any saved run bytes."""
import hashlib
import json

import pytest
import torch

from hypergan.checkpoints import read_checkpoint
from hypergan.config import write_default
from hypergan.execution import train
from hypergan.signal_diagnostic import diagnose


def _files(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def checkpoint_run(tmp_path):
    config = write_default(tmp_path / 'project', device='cpu')
    root = tmp_path / 'run'
    train(config, root, steps=2, stop_after_steps=1)
    return config, root


def test_checkpoint_probe_loads_saved_online_pair_and_preserves_every_run_file(checkpoint_run, monkeypatch):
    import hypergan.signal_diagnostic as diagnostic
    _, root = checkpoint_run
    target, metadata, state = read_checkpoint(root)
    before = _files(root)
    original_probe = diagnostic._probe
    def verified_probe(trainer, objective):
        assert trainer.step == 1
        assert not trainer.opt_g.state and not trainer.opt_d.state
        assert not hasattr(trainer, 'ema_graph')
        for name in ('graph', 'prior'):
            module = getattr(trainer, name)
            for key, value in module.state_dict().items():
                assert torch.equal(value.cpu(), state[name][key])
            for key, value in module.named_buffers():
                assert torch.equal(value.cpu(), state['buffers'][name][key])
        assert any(not torch.equal(value, state['ema_graph'][name])
                   for name, value in state['graph'].items() if value.is_floating_point())
        return original_probe(trainer, objective)
    monkeypatch.setattr(diagnostic, '_probe', verified_probe)
    result = diagnose(root)
    assert _files(root) == before
    assert result['kind'] == 'generator-checkpoint-signal'
    assert result['phase'] == 'frozen-checkpoint-generator-and-discriminator-at-step-1'
    assert result['step'] == result['checkpoint']['step'] == 1
    assert result['checkpoint']['path'] == str(target)
    assert result['checkpoint']['state_sha256'] == metadata['state_sha256']
    protocol = result['checkpoint']['diagnostic_protocol']
    assert protocol['optimizer_steps'] == 0 and not protocol['optimizer_state_loaded']
    assert set(protocol['named_rng_streams'].values()) == {'restored'}
    assert result['state_verification']['optimizer_steps'] == 0


def test_explicit_older_checkpoint_measures_its_weights_not_latest(checkpoint_run):
    _, root = checkpoint_run
    initial = next(path for path in (root / 'checkpoints').iterdir()
                   if path.is_dir() and json.loads((path / 'manifest.json').read_text())['step'] == 0)
    before = _files(root)
    latest = diagnose(root)
    old = diagnose(root, checkpoint=initial.name)
    assert old['step'] == 0 and latest['step'] == 1
    assert old['checkpoint']['state_sha256'] != latest['checkpoint']['state_sha256']
    assert old['parameters'] != latest['parameters']
    assert _files(root) == before


def test_checkpoint_probe_batch_and_device_override_records_effective_protocol(checkpoint_run):
    _, root = checkpoint_run
    before = _files(root)
    result = diagnose(root, device='cpu', batch_size=2)
    assert result['configured_batch_size'] != result['probe_batch_size'] == 2
    assert result['effective_config_fingerprint'] != result['config_fingerprint']
    assert result['checkpoint']['diagnostic_protocol']['source_runtime']['device'] == 'cpu'
    assert result['checkpoint']['diagnostic_protocol']['evaluation_runtime']['device'] == 'cpu'
    assert _files(root) == before


def test_checkpoint_integrity_failure_precedes_probe(checkpoint_run, monkeypatch):
    import hypergan.signal_diagnostic as diagnostic
    _, root = checkpoint_run
    target, _, _ = read_checkpoint(root)
    state = target / 'state.pt'
    state.write_bytes(state.read_bytes() + b'corrupt')
    before = _files(root)
    monkeypatch.setattr(diagnostic, '_probe', lambda *a, **k: pytest.fail('corrupt checkpoint was probed'))
    with pytest.raises(ValueError, match='digest mismatch'):
        diagnose(root)
    assert _files(root) == before


def test_checkpoint_selection_rejects_foreign_and_inference_sources(checkpoint_run, tmp_path):
    config, root = checkpoint_run
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    with pytest.raises(ValueError, match='inside this run'):
        diagnose(root, checkpoint=foreign)
    with pytest.raises(ValueError, match='full online generator and discriminator'):
        diagnose(root / 'model.pt')
    with pytest.raises(ValueError, match='requires a run directory'):
        diagnose(config, checkpoint='latest')


def test_initial_config_diagnostic_still_reports_initialization(checkpoint_run):
    config, _ = checkpoint_run
    result = diagnose(config, batch_size=2)
    assert result['kind'] == 'generator-initial-signal' and result['step'] == 0
    assert result['config_path'] == str(config)
    assert 'checkpoint' not in result
