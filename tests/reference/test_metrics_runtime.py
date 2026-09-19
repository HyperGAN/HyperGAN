"""Scalar publication is independent of complete numerical state and recovery."""
import json
from pathlib import Path

import pytest
import torch

from hypergan.checkpoints import read_checkpoint
from hypergan.config import write_default
from hypergan.metrics import read_catalog
from hypergan.training import train, resume


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def events(run):
    return [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]


def config(tmp_path, name, metrics=''):
    path = write_default(tmp_path / name, device='cpu')
    path.write_text(path.read_text().replace('num_particles = 20000', 'num_particles = 32') + metrics)
    return path


def test_metrics_on_off_cadence_and_changed_resume_are_numerically_identical(tmp_path):
    on = config(tmp_path, 'on-config')
    off = config(tmp_path, 'off-config', '\n[metrics]\npreset = "none"\n')
    sparse = config(tmp_path, 'sparse-config', '\n[metrics]\nevery_steps = 2\ndisable = ["loss/d_total"]\n')
    train(on, tmp_path / 'on')
    train(off, tmp_path / 'off')
    train(sparse, tmp_path / 'sparse')
    for name in ('off', 'sparse'):
        same(read_checkpoint(tmp_path / 'on')[2], read_checkpoint(tmp_path / name)[2])
    for row in events(tmp_path / 'off'):
        assert 'd_loss' not in row and 'g_loss' not in row
        if row['event'] == 'train':
            assert row['metrics'] == {} and row['measurement_status'] == {}
    for row in events(tmp_path / 'sparse'):
        if row['event'] == 'train':
            assert bool(row['metrics']) == (row['step'] % 2 == 0)
            assert 'loss/d_total' not in row['metrics']
    stopped = train(on, tmp_path / 'resumed', checkpoint_every=1, stop_after_steps=2)
    old_revision = stopped['metrics_catalog']
    old_catalog = read_catalog(tmp_path / 'resumed', old_revision)
    completed = resume(tmp_path / 'resumed', config_path=off)
    assert old_revision != completed['metrics_catalog']
    assert read_catalog(tmp_path / 'resumed', old_revision) == old_catalog
    assert read_catalog(tmp_path / 'resumed')['metrics'] == {}
    same(read_checkpoint(tmp_path / 'on')[2], read_checkpoint(tmp_path / 'resumed')[2])
    zero = resume(tmp_path / 'resumed', checkpoint=stopped['checkpoint_path'], max_seconds=1e-12, config_path=sparse)
    assert zero['steps'] == 2
    lineage = next(row for row in events(tmp_path / 'resumed') if row['attempt_id'] == zero['attempt_id'] and row['event'] == 'resume')
    assert lineage['parent_attempt_id'] == stopped['attempt_id'] and lineage['restored_step'] == 2
    assert lineage['checkpoint_id'] == Path(stopped['checkpoint_path']).name
    assert len(lineage['checkpoint_sha256']) == 64
    assert not [row for row in events(tmp_path / 'resumed') if row['attempt_id'] == zero['attempt_id'] and row['event'] == 'train']
    resume(tmp_path / 'resumed')
    same(read_checkpoint(tmp_path / 'on')[2], read_checkpoint(tmp_path / 'resumed')[2])


def test_named_loss_decomposition_and_half_update_never_publishes(tmp_path, monkeypatch):
    path = config(tmp_path, 'recipe')
    path.write_text(path.read_text().replace('weight = 1.0', 'weight = 0.25', 1).replace('lazy_k = 1', 'lazy_k = 2') +
        '\n[[objectives]]\nid = "reconstruction"\nfactory = "mse"\nweight = 0.5\ninputs = { input = "generated", target = "batch.real" }\n')
    train(path, tmp_path / 'on')
    for row in events(tmp_path / 'on'):
        if row['event'] != 'train':
            continue
        m = row['metrics']
        assert m['loss/total'] == pytest.approx(m['loss/d_total'] + m['loss/g_total'])
        assert m['loss/d_total'] == pytest.approx(m['loss/d_adversarial'] + m['loss/gradient_penalty'])
        assert m['loss/g_total'] == pytest.approx(m['loss/g_adversarial'] + m['loss/prior_regularizer'] + m['loss/objectives/reconstruction'])
        assert m['loss/d_adversarial'] == m['loss/d_adversarial_raw'] * .25
        assert row['measurement_status']['loss/gradient_penalty']['applied'] == (row['step'] % 2 == 0)
    from hypergan.training import DeviceAdam
    original = DeviceAdam.step
    calls = 0
    def fail(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected half update')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(DeviceAdam, 'step', fail)
    with pytest.raises(RuntimeError, match='half update'):
        train(path, tmp_path / 'failed')
    assert not [row for row in events(tmp_path / 'failed') if row['event'] == 'train']
    assert read_checkpoint(tmp_path / 'failed')[2]['step'] == 0


def test_disabling_metrics_does_not_disable_completion_validation(tmp_path, monkeypatch):
    from hypergan.training import ReferenceTrainer
    path = config(tmp_path, 'disabled', '\n[metrics]\npreset = "none"\n')
    original = ReferenceTrainer.update
    def invalid(self, *args, **kwargs):
        row, batch = original(self, *args, **kwargs)
        row['g_adversarial'] = float('nan')
        return row, batch
    monkeypatch.setattr(ReferenceTrainer, 'update', invalid)
    with pytest.raises(ValueError, match='finite'):
        train(path, tmp_path / 'failed')
    assert not [row for row in events(tmp_path / 'failed') if row['event'] == 'train']
    assert read_checkpoint(tmp_path / 'failed')[2]['step'] == 0
