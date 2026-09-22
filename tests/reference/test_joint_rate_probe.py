"""The disposable research runner persists comparable, isolated measurements."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from hypergan.config import write_default
from hypergan.checkpoints import trainer_state
from hypergan.startup_dynamics import _same_state, _snapshot

_spec = importlib.util.spec_from_file_location(
    'joint_rate_probe', Path(__file__).resolve().parents[2] / 'reports/joint_rate_probe.py')
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def config_path(tmp_path):
    path = write_default(tmp_path / 'config', device='cpu')
    path.write_text(path.read_text().replace('steps = 5', 'steps = 512')
                    .replace('num_particles = 20000', 'num_particles = 32'))
    return path


def test_milestones_include_initial_final_and_support_long_rollout():
    assert probe._observation_steps(512) == [0, 1, 8, 16, 32, 64, 128, 256, 512]
    assert probe._observation_steps(350, [8, 350, 512, 8]) == [0, 8, 350]
    for values in ([True], [-1], [513], [.5]):
        with pytest.raises(ValueError):
            probe._observation_steps(32, values)


def test_atomic_report_preserves_old_complete_json_on_encoding_failure(tmp_path):
    destination = tmp_path / 'progress.json'
    probe._atomic_report(destination, {'step': 0})
    with pytest.raises(ValueError):
        probe._atomic_report(destination, {'step': float('nan')})
    assert json.loads(destination.read_text()) == {'step': 0}
    assert list(tmp_path.iterdir()) == [destination]
    probe._atomic_report(destination, {'step': 8})
    assert json.loads(destination.read_text()) == {'step': 8}


def test_native_rollout_progress_and_protocol_exclude_unrequested_probes(tmp_path, monkeypatch):
    path = config_path(tmp_path)
    original = path.read_bytes()
    destination = tmp_path / 'progress.json'
    snapshots = []
    write = probe._atomic_report
    def observe_write(destination, report):
        write(destination, report)
        snapshots.append(json.loads(destination.read_text()))
    monkeypatch.setattr(probe, '_atomic_report', observe_write)
    def forbidden_observer(*args, **kwargs):
        pytest.fail('Disabled direction diagnostics constructed an observer')
    monkeypatch.setattr(probe, 'UpdateObserver', forbidden_observer)
    report = probe.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=2,
                             observe_steps=[0, 2], progress_path=destination)
    assert report['status'] == 'complete' and report['completed_updates'] == 2
    assert report['restored'] and report['source_config_unchanged']
    assert path.read_bytes() == original
    assert [row['completed_updates'] for row in snapshots] == [0, 2, 2]
    assert [row['status'] for row in snapshots] == ['running', 'running', 'complete']
    assert snapshots[-1] == report
    assert report['budget']['rollout_observation_generator_forwards'] == 4
    assert 'g_response_forwards' not in report['budget']
    assert 'crossed_progress' not in report and 'directional_probes' not in report
    assert report['timings']['training_update_seconds'] > 0
    assert report['timings']['diagnostic_seconds'] > 0
    assert report['timings']['observer_seconds_inside_update'] == 0
    assert report['evaluation']['configured_training_horizon'] == 512
    assert report['evaluation']['execution_backend'] == 'native'
    assert report['evaluation']['device_type'] == 'cpu'
    assert report['evaluation']['torch_version'] == str(torch.__version__)
    assert report['evaluation']['batch_size'] > 0
    assert report['evaluation']['device_hardware']
    assert report['initial_parameters_sha256'] == report['prepared_parameters_sha256']
    assert report['real_bank_output_stats']['status'] == 'finite'
    initial_timing, final_timing = [row['timing_at_observation'] for row in report['observations']]
    assert initial_timing['training_update_seconds'] == 0
    assert final_timing['training_update_seconds'] > 0
    assert final_timing['elapsed_seconds'] > initial_timing['elapsed_seconds']
    for key in ('bank_sha256', 'measurement_rng_sha256', 'prior_rng_sha256'):
        assert len(report['evaluation'][key]) == 64
    with pytest.raises(FileExistsError):
        probe.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=2, progress_path=destination)
    assert json.loads(destination.read_text()) == report


def test_prepare_group_layout_is_closed_before_exact_state_restore(tmp_path):
    path = config_path(tmp_path)
    captured = {}
    @contextmanager
    def prepare(trainer):
        captured['trainer'] = trainer
        # Capture original state through the trainer factory below, before supplied rates.
        groups = list(trainer.opt_g.param_groups)
        bases = list(trainer.base_lrs[0])
        first = groups[0]
        params = first['params']
        assert len(params) > 1
        trainer.opt_g.param_groups = [{**first, 'params': params[:1]},
                                     {**first, 'params': params[1:], 'lr': first['lr'] * .5},
                                     *groups[1:]]
        trainer.base_lrs[0] = [bases[0], bases[0] * .5, *bases[1:]]
        try:
            yield {'algorithm': 'test-group-split'}
        finally:
            trainer.opt_g.param_groups = groups
            trainer.base_lrs[0] = bases
            captured['closed'] = True
    factory = probe.ReferenceTrainer
    def record(config):
        trainer = factory(config)
        captured['initial'] = _snapshot(trainer)
        return trainer
    from unittest.mock import patch
    with patch.object(probe, 'ReferenceTrainer', record):
        report = probe.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=1, prepare=prepare)
    assert report['status'] == 'complete', report.get('failure')
    assert captured['closed'] and report['restored']
    assert _same_state(captured['initial']['state'], trainer_state(captured['trainer'], None))
    assert report['proposal']['algorithm'] == 'test-group-split'
    assert len(report['per_step'][0]['actual_lrs'][0]) == len(report['original_base_lrs'][0]) + 1
    assert report['timings']['preparation_seconds'] > 0


def test_failed_prepare_restores_weights_and_persists_failure(tmp_path):
    path = config_path(tmp_path)
    captured = {}
    @contextmanager
    def prepare(trainer):
        captured['trainer'] = trainer
        with torch.no_grad():
            next(trainer.graph.models['generator'].parameters()).add_(3.)
        raise RuntimeError('deliberate proposal failure')
        yield {}
    report = probe.run_probe(path, g_lr=1e-4, d_lr=1e-4, steps=1,
                             prepare=prepare, progress_path=tmp_path / 'failed.json')
    assert report['status'] == 'failed' and report['restored']
    assert report['failure']['message'] == 'deliberate proposal failure'
    assert report['completed_updates'] == 0
    assert json.loads((tmp_path / 'failed.json').read_text()) == report
