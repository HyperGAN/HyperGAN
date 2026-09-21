"""Exact CPU continuation and failure-boundary contract."""
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from hypergan.checkpoints import read_checkpoint, trainer_state, restore_trainer
from hypergan.config import resolve_config, write_default
from hypergan.run_state import read_events, run_lock
from hypergan.training import ReferenceTrainer, resume, train


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_public_train_repeats_resume_exact_state_and_completed_schedule(tmp_path):
    from hypergan.execution import train as public_train
    config = write_default(tmp_path / 'config', device='cpu')
    full, split = tmp_path / 'full', tmp_path / 'split'
    public_train(config, full, steps=4, checkpoint_every=1)
    stopped = public_train(config, split, steps=4, checkpoint_every=1, stop_after_steps=2)
    assert stopped['status'] == 'stopped' and stopped['last_durable_step'] == 2
    old_sample = Path(stopped['sample_path']).read_bytes()
    old_bundle = Path(stopped['bundle_path']).read_bytes()
    resumed = public_train(config, split, steps=4)
    assert resumed['status'] == 'complete' and resumed['steps'] == 4
    assert resumed['attempt_index'] == 2
    assert resumed['run_id'] == stopped['run_id']
    assert resumed['checkpoint_every'] == 1
    equal(read_checkpoint(full)[2], read_checkpoint(split)[2])
    assert Path(stopped['sample_path']).read_bytes() == old_sample
    assert Path(stopped['bundle_path']).read_bytes() == old_bundle

    before = {p.relative_to(split): p.read_bytes() for p in split.rglob('*') if p.is_file()}
    for steps in (None, 3, 5):
        with pytest.raises(ValueError, match='configuration differs'):
            public_train(config, split, steps=steps)
        assert before == {p.relative_to(split): p.read_bytes() for p in split.rglob('*') if p.is_file()}

    events = []
    repeated = public_train(config, split, steps=4, on_event=events.append)
    assert repeated['status'] == 'complete' and repeated['steps'] == 4
    assert repeated['run_id'] == stopped['run_id']
    assert not [row for row in events if row['event'] == 'train']
    equal(read_checkpoint(full)[2], read_checkpoint(split)[2])


def test_repeated_train_checks_metric_argument_types_under_lock_before_factories(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from hypergan.execution import prepare_train
    import hypergan.run_controller as controller

    config = write_default(tmp_path / 'config', device='cpu')
    config.write_text(config.read_text() + '''
[metrics]
disable = ["custom/typed"]
[metrics.custom."custom/typed"]
factory = "uninstalled.metrics:Typed"
inputs = { value = "update.g_loss" }
args = { flag = true }
''')
    run = tmp_path / 'run'
    train(config, run, stop_after_steps=1)
    prepared = prepare_train(config, run)
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    original_lock = controller.run_lock

    @contextmanager
    def change_config_after_preparation(run_dir):
        with original_lock(run_dir):
            config.write_text(config.read_text().replace('flag = true', 'flag = 1'))
            yield

    monkeypatch.setattr(controller, 'run_lock', change_config_after_preparation)
    monkeypatch.setattr(controller, 'prepare_custom',
                        lambda _: pytest.fail('custom metric factory validation preceded config rejection'))
    with pytest.raises(ValueError, match='configuration differs'):
        prepared.run()
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


@pytest.mark.heavy
def test_resume_matches_uninterrupted_and_replays_old_checkpoint_without_overwrite(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    full = train(config, tmp_path / 'full', checkpoint_every=1)
    stopped = train(config, tmp_path / 'split', checkpoint_every=1, stop_after_steps=3)
    assert stopped['status'] == 'stopped' and stopped['steps'] == stopped['last_durable_step'] == 3
    old_checkpoint = stopped['checkpoint_path']
    old_sample = Path(stopped['sample_path']).read_bytes()
    old_bundle = Path(stopped['bundle_path']).read_bytes()
    # Exercise a genuinely fresh interpreter with only saved state available.
    subprocess.run([sys.executable, '-c', 'from hypergan.training import resume; import sys; resume(sys.argv[1])', str(tmp_path / 'split')], check=True, cwd=tmp_path)
    completed = json.loads((tmp_path / 'split/manifest.json').read_text())
    assert completed['status'] == 'complete' and completed['attempt_index'] == 2
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'split')[2])
    replay = resume(tmp_path / 'split', checkpoint=old_checkpoint)
    assert replay['attempt_index'] == 3
    assert len({stopped['sample_path'], completed['sample_path'], replay['sample_path']}) == 3
    assert Path(stopped['sample_path']).read_bytes() == old_sample
    assert Path(stopped['bundle_path']).read_bytes() == old_bundle
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'split')[2])


def test_failure_after_discriminator_update_preserves_prior_durable_boundary(tmp_path, monkeypatch):
    config = write_default(tmp_path / 'config', device="cpu")
    full = train(config, tmp_path / 'full')
    from hypergan.training import DeviceAdam
    original = DeviceAdam.step
    calls = 0
    def fail_generator(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('after D, before G')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(DeviceAdam, 'step', fail_generator)
    with pytest.raises(KeyboardInterrupt):
        train(config, tmp_path / 'interrupted', checkpoint_every=1)
    manifest = json.loads((tmp_path / 'interrupted/manifest.json').read_text())
    assert manifest['status'] == 'interrupted' and manifest['steps'] == manifest['last_durable_step'] == 0
    assert read_checkpoint(tmp_path / 'interrupted')[2]['step'] == 0
    monkeypatch.setattr(DeviceAdam, 'step', original)
    resume(tmp_path / 'interrupted')
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'interrupted')[2])


def test_runtime_config_tampering_and_inference_rejected_without_mutation(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    stopped = train(config, tmp_path / 'run', stop_after_steps=2)
    manifest_path = tmp_path / 'run/manifest.json'
    before = manifest_path.read_bytes()
    other = write_default(tmp_path / 'other', device="cpu")
    other.write_text(other.read_text().replace('steps = 5', 'steps = 6'))
    with pytest.raises(ValueError, match='configuration'):
        resume(tmp_path / 'run', config_path=other)
    with pytest.raises(ValueError, match='completed checkpoint'):
        resume(tmp_path / 'run', checkpoint=stopped['bundle_path'])
    assert manifest_path.read_bytes() == before
    info = Path(stopped['checkpoint_path']) / 'manifest.json'
    value = json.loads(info.read_text())
    value['runtime']['world_size'] = 2
    info.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='runtime/topology'):
        resume(tmp_path / 'run')
    assert manifest_path.read_bytes() == before


def test_checkpoint_write_failure_keeps_last_durable_pointer(tmp_path, monkeypatch):
    config = write_default(tmp_path / 'config', device="cpu")
    original = torch.save
    calls = 0
    def broken(value, destination, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.write(b'partial')
            raise OSError('disk full')
        return original(value, destination, *args, **kwargs)
    monkeypatch.setattr(torch, 'save', broken)
    with pytest.raises(OSError, match='disk full'):
        train(config, tmp_path / 'run', checkpoint_every=1)
    manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
    assert manifest['steps'] == 1 and manifest['last_durable_step'] == 0
    assert manifest['possible_lost_steps'] == 1
    assert read_checkpoint(tmp_path / 'run')[2]['step'] == 0
    assert not list((tmp_path / 'run/checkpoints').glob('.pending-*'))


def test_global_rng_restore_after_constructors_and_module_modes():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = resolve_config({})
        first = ReferenceTrainer(config)
        initial = (random.random(), float(np.random.random()), torch.rand(1))
        ReferenceTrainer(config)
        equal(initial, (random.random(), float(np.random.random()), torch.rand(1)))
        first.graph.models['generator'].eval()
        row, batch = first.update()
        state = trainer_state(first, batch)
        expected = (random.random(), float(np.random.random()), torch.rand(1))
        second = ReferenceTrainer(config)
        restore_trainer(second, state)
        equal(expected, (random.random(), float(np.random.random()), torch.rand(1)))
        assert not second.graph.models['generator'].training
        equal(first.base_lrs, second.base_lrs)
    finally:
        torch.set_num_threads(previous)


def test_lock_partial_log_and_observer_error_are_honest(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    def observer(row):
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError('observer offline')
    with pytest.warns(RuntimeWarning, match='observer failed'):
        stopped = train(config, tmp_path / 'run', stop_after_steps=1, on_event=observer)
    with run_lock(tmp_path / 'run'):
        with pytest.raises(RuntimeError, match='locked'):
            resume(tmp_path / 'run')
    with (tmp_path / 'run/events.jsonl').open('ab') as output:
        output.write(b'{"event":"partial')
    assert read_events(tmp_path / 'run')[-1]['event'] == 'stopped'
    resume(tmp_path / 'run')
    rows = read_events(tmp_path / 'run', limit=3)
    assert len(rows) == 3 and rows[-1]['event'] == 'complete'
    assert all(row['schema_version'] == 2 for row in rows)


def test_wall_time_zero_update_stop_has_checkpoint_no_inference(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    result = train(config, tmp_path / 'run', max_seconds=1e-12)
    assert result['status'] == 'stopped' and result['steps'] == result['last_durable_step'] == 0
    assert result['stop_reason'] == 'max_seconds' and 'sample_path' not in result
    assert resume(tmp_path / 'run')['status'] == 'complete'


def test_nonpersistent_buffers_and_dynamic_trainability_restored():
    config = resolve_config({})
    first = ReferenceTrainer(config)
    second = ReferenceTrainer(config)
    for trainer in (first, second):
        trainer.graph.models['generator'].register_buffer('ephemeral', torch.zeros(2), persistent=False)
    first.graph.models['generator'].ephemeral.add_(3)
    next(first.graph.models['generator'].parameters()).requires_grad_(False)
    state = trainer_state(first, None)
    restore_trainer(second, state)
    assert torch.equal(second.graph.models['generator'].ephemeral, torch.full((2,), 3.0))
    assert not next(second.graph.models['generator'].parameters()).requires_grad


@pytest.mark.parametrize('damage', ['metadata-list', 'metadata-key', 'payload', 'missing-optimizer'])
def test_malformed_checkpoint_is_actionable_without_run_mutation(tmp_path, damage):
    import hashlib
    config = write_default(tmp_path / 'config', device="cpu")
    manifest = train(config, tmp_path / 'run', stop_after_steps=1)
    path = Path(manifest['checkpoint_path'])
    run_manifest = (tmp_path / 'run/manifest.json').read_bytes()
    metadata = json.loads((path / 'manifest.json').read_text())
    if damage == 'metadata-list':
        metadata = []
    elif damage == 'metadata-key':
        del metadata['state_sha256']
    elif damage == 'payload':
        (path / 'state.pt').write_bytes(b'not a tensor checkpoint')
        metadata['state_sha256'] = hashlib.sha256((path / 'state.pt').read_bytes()).hexdigest()
    else:
        state = torch.load(path / 'state.pt', weights_only=True)
        state['optimizers'].pop()
        torch.save(state, path / 'state.pt')
        metadata['state_sha256'] = hashlib.sha256((path / 'state.pt').read_bytes()).hexdigest()
    (path / 'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='[Cc]heckpoint'):
        resume(tmp_path / 'run')
    assert (tmp_path / 'run/manifest.json').read_bytes() == run_manifest


def test_older_checkpoint_zero_update_resume_updates_default_pointer(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    stopped = train(config, tmp_path / 'run', stop_after_steps=2)
    resume(tmp_path / 'run')
    rolled_back = resume(tmp_path / 'run', checkpoint=stopped['checkpoint_path'], max_seconds=1e-12)
    assert rolled_back['steps'] == rolled_back['last_durable_step'] == 2
    assert read_checkpoint(tmp_path / 'run')[2]['step'] == 2
    continued = resume(tmp_path / 'run', stop_after_steps=1)
    assert continued['steps'] == 3
    assert continued['next_sample_sequence'] > rolled_back['next_sample_sequence'] > stopped['next_sample_sequence']


def test_observer_rng_and_failure_do_not_change_numerics(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    train(config, tmp_path / 'normal')
    def callback(row):
        random.random()
        np.random.random()
        torch.rand(100)
        raise RuntimeError('observer unavailable')
    with pytest.warns(RuntimeWarning, match='observer failed'):
        train(config, tmp_path / 'observed', on_event=callback)
    equal(read_checkpoint(tmp_path / 'normal')[2], read_checkpoint(tmp_path / 'observed')[2])


def test_unknown_data_trains_without_false_recovery_claim(tmp_path, monkeypatch):
    import hypergan.training as module
    class UnknownData:
        def __call__(self, batch_size, *, generator):
            return {'real': torch.randn(batch_size, 2, generator=generator)}
    import types
    monkeypatch.setitem(sys.modules, 'unknown', types.ModuleType('unknown'))
    original = module.construct
    monkeypatch.setattr(module, 'construct', lambda spec: UnknownData() if spec['factory'] == 'unknown:Data' else original(spec))
    config = write_default(tmp_path / 'config', device="cpu")
    config.write_text(config.read_text().replace('factory = "gaussian_grid"', 'factory = "unknown:Data"'))
    with pytest.warns(RuntimeWarning, match='Custom data'):
        result = train(config, tmp_path / 'run')
    assert result['status'] == 'complete' and not result['resume_supported']
    assert result['last_durable_step'] is None and result['checkpoint_path'] is None
    with pytest.raises(ValueError, match='No full training checkpoint'):
        resume(tmp_path / 'run')
