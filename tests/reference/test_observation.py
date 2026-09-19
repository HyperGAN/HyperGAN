"""Observation never changes numerical state or publishes partial artifacts."""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from hypergan.checkpoints import read_checkpoint, trainer_state
from hypergan.config import load_config, write_default
from hypergan.previews import MAX_BYTES, MAX_COUNT, render_preview
from hypergan.training import ReferenceTrainer, resume, train


class StochasticGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)
        self.drop = nn.Dropout(0.3)
        self.register_buffer('counter', torch.zeros(()), persistent=False)

    def forward(self, x):
        self.counter.add_(1)
        value = self.drop(self.bn(self.linear(x)))
        noise = torch.rand_like(value) + random.random() + float(np.random.random())
        return value + 0.001 * noise + self.counter * 0.001


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def stochastic_config(path):
    path.write_text(f'''
[components.generator]
factory = "{__name__}:StochasticGenerator"
inputs = {{x = "latent"}}
[components.discriminator]
factory = "linear"
args = {{in_features = 2, out_features = 1}}
inputs = {{input = "candidate"}}
[prior.args]
num_particles = 32
z_dim = 4
[training]
steps = 6
batch_size = 4
[sampling]
count = 20
seed = 9
''')
    return path


def test_preview_schedule_retention_and_resume_preserve_complete_state(tmp_path):
    config = stochastic_config(tmp_path / 'config.toml')
    full = train(config, tmp_path / 'full')
    stopped = train(config, tmp_path / 'observed', preview_every=1, preview_keep=2, stop_after_steps=3)
    assert len(stopped['previews']) == 2
    initial_paths = {record['path'] for record in stopped['previews']}
    initial_sequence = stopped['next_sample_sequence']
    done = resume(tmp_path / 'observed')
    assert done['preview_every'] == 1 and done['preview_keep'] == 2
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'observed')[2])
    assert done['next_sample_sequence'] > initial_sequence
    index = json.loads((tmp_path / 'observed/previews/index.json').read_text())
    assert [record['step'] for record in index['previews']] == [5, 6]
    assert not any(Path(path).exists() for path in initial_paths)
    directories = [path for path in (tmp_path / 'observed/previews').iterdir() if path.is_dir()]
    assert len(directories) == 2
    for record in index['previews']:
        path = Path(record['path'])
        payload = json.loads(path.read_text())
        assert path.stat().st_size == record['bytes'] <= MAX_BYTES
        assert payload['count'] == MAX_COUNT and payload['identity']['sample_sequence'] < done['next_sample_sequence']
    # Retention applies only to managed periodic previews.
    assert Path(stopped['bundle_path']).exists() and Path(stopped['sample_path']).exists()
    preserved = {record['path']: Path(record['path']).read_bytes() for record in done['previews']}
    replay = resume(tmp_path / 'observed', checkpoint=stopped['checkpoint_path'], preview_every=0, stop_after_steps=1)
    assert replay['preview_every'] == 0 and replay['next_sample_sequence'] > done['next_sample_sequence']
    assert all(Path(path).read_bytes() == content for path, content in preserved.items())


def test_direct_preview_preserves_ema_live_modes_buffers_rng_and_conditioning(tmp_path):
    config = stochastic_config(tmp_path / 'config.toml')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer = ReferenceTrainer(load_config(config))
        _, batch = trainer.update()
        trainer.ema_graph.models['generator'].train()
        state = copy.deepcopy(trainer_state(trainer, batch))
        identity = {'run_id': 'run', 'attempt_id': 'attempt', 'sample_sequence': 1}
        a = render_preview(trainer, batch, identity)
        b = render_preview(trainer, batch, identity)
        assert a == b
        equal(state, trainer_state(trainer, batch))
    finally:
        torch.set_num_threads(previous)


def test_preview_write_failure_is_observer_only_and_sequence_not_reused(tmp_path, monkeypatch):
    import hypergan.previews as previews
    config = write_default(tmp_path / 'config', device="cpu")
    train(config, tmp_path / 'full')
    def disk_full(*args, **kwargs):
        raise OSError('preview volume full')
    monkeypatch.setattr(previews, '_write_bounded', disk_full)
    result = train(config, tmp_path / 'observed', preview_every=1)
    assert result['status'] == 'complete' and result['next_sample_sequence'] == 7
    assert len(result['observation_errors']) == 5
    assert all(error['source'] == 'preview' for error in result['observation_errors'])
    assert not [path for path in (tmp_path / 'observed/previews').iterdir() if path.is_dir()]
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'observed')[2])


def test_preview_byte_limit_is_enforced_before_publication(tmp_path, monkeypatch):
    import hypergan.previews as previews
    config = write_default(tmp_path / 'config', device="cpu")
    monkeypatch.setattr(previews, 'MAX_BYTES', 80)
    result = train(config, tmp_path / 'run', preview_every=1)
    assert result['status'] == 'complete' and not result['previews']
    assert all('byte budget' in error['error'] for error in result['observation_errors'])


def test_manual_requests_coalesce_acknowledge_durable_checkpoint(tmp_path):
    from hypergan.run_requests import submit_checkpoint_request, checkpoint_request_status
    config = write_default(tmp_path / 'config', device="cpu")
    requests = []
    def observer(row):
        if row['event'] == 'train' and row['step'] == 2:
            for _ in range(2):
                request = submit_checkpoint_request(tmp_path / 'run', run_id=row['run_id'], attempt_id=row['attempt_id'])
                requests.append(request['request']['request_id'])
    result = train(config, tmp_path / 'run', checkpoint_every=100, on_event=observer)
    receipts = [checkpoint_request_status(tmp_path / 'run', request_id) for request_id in requests]
    assert len(receipts) == 2
    assert all(receipt['status'] == 'succeeded' and receipt['step'] == 2 for receipt in receipts)
    assert receipts[0]['checkpoint_path'] == receipts[1]['checkpoint_path']
    _, metadata, state = read_checkpoint(tmp_path / 'run', receipts[0]['checkpoint_path'])
    assert set(metadata['request_ids']) == set(requests) and state['step'] == 2
    assert result['status'] == 'complete'


def test_manual_request_is_never_serviced_after_partial_update(tmp_path, monkeypatch):
    from hypergan.run_requests import submit_checkpoint_request, checkpoint_request_status
    config = write_default(tmp_path / 'config', device="cpu")
    requests = []
    original = torch.optim.Adam.step
    calls = 0
    def fail_generator(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
            request = submit_checkpoint_request(tmp_path / 'run', run_id=manifest['run_id'], attempt_id=manifest['attempt_id'])
            requests.append(request['request']['request_id'])
            raise KeyboardInterrupt('half update')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', fail_generator)
    with pytest.raises(KeyboardInterrupt):
        train(config, tmp_path / 'run')
    assert read_checkpoint(tmp_path / 'run')[2]['step'] == 0
    assert checkpoint_request_status(tmp_path / 'run', requests[0])['status'] == 'pending'
    monkeypatch.setattr(torch.optim.Adam, 'step', original)
    resume(tmp_path / 'run')
    receipt = checkpoint_request_status(tmp_path / 'run', requests[0])
    assert receipt['status'] == 'rejected' and 'attempt' in receipt['error']


class MutatingConditionGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x, condition):
        condition.add_(2)
        return self.linear(x) + condition


def test_preview_clones_real_conditioning_before_custom_forward(tmp_path):
    from hypergan.config import resolve_config
    from hypergan.config import DEFAULT
    raw = copy.deepcopy(DEFAULT)
    raw['components']['generator'] = {'factory': f'{__name__}:MutatingConditionGenerator',
        'inputs': {'x': 'latent', 'condition': 'batch.condition'}}
    config = resolve_config(raw)
    trainer = ReferenceTrainer(config)
    batch = {'real': torch.zeros(4, 2), 'condition': torch.ones(4, 2)}
    before = copy.deepcopy(batch)
    payload = render_preview(trainer, batch, {'run_id': 'run', 'attempt_id': 'attempt', 'sample_sequence': 1})
    assert payload['conditioning'] == 'last-completed-batch-cycled'
    assert payload['inputs']['condition'] == [[1.0, 1.0]] * payload['count']
    equal(before, batch)


def test_killed_pending_preview_is_cleaned_without_touching_unmanaged_files(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    stopped = train(config, tmp_path / 'run', stop_after_steps=1)
    root = tmp_path / 'run/previews'
    root.mkdir()
    pending = root / ('.pending-000000000001-' + stopped['attempt_id'] + '-step00000001-' + 'a' * 32)
    pending.mkdir()
    (pending / 'preview.json').write_text('partial')
    unmanaged = root / '.pending-user-notes'
    unmanaged.mkdir()
    (unmanaged / 'notes').write_text('keep')
    result = resume(tmp_path / 'run', preview_every=1, preview_keep=1)
    assert not pending.exists() and (unmanaged / 'notes').read_text() == 'keep'
    assert len(result['previews']) == 1 and result['previews'][0]['step'] == 5


def test_failed_index_write_does_not_accumulate_published_orphans(tmp_path, monkeypatch):
    import hypergan.previews as previews
    original = previews.atomic_json
    def fail_index(path, value):
        if Path(path).name == 'index.json':
            raise OSError('index write failed')
        return original(path, value)
    monkeypatch.setattr(previews, 'atomic_json', fail_index)
    config = write_default(tmp_path / 'config', device="cpu")
    result = train(config, tmp_path / 'run', preview_every=1)
    assert result['status'] == 'complete' and not result['previews']
    assert not [path for path in (tmp_path / 'run/previews').iterdir() if path.is_dir()]


def test_acknowledgement_retry_reuses_identifiable_saved_checkpoint(tmp_path, monkeypatch):
    import hypergan.run_requests as requests_api
    original = requests_api.acknowledge_request
    attempts = 0
    def transient_failure(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError('receipt unavailable once')
        return original(*args, **kwargs)
    monkeypatch.setattr(requests_api, 'acknowledge_request', transient_failure)
    ids = []
    def observer(row):
        if row['event'] == 'train' and row['step'] == 1:
            result = requests_api.submit_checkpoint_request(tmp_path / 'run', run_id=row['run_id'], attempt_id=row['attempt_id'])
            ids.append(result['request']['request_id'])
    config = write_default(tmp_path / 'config', device="cpu")
    train(config, tmp_path / 'run', on_event=observer, checkpoint_every=100)
    receipt = requests_api.checkpoint_request_status(tmp_path / 'run', ids[0])
    assert receipt['status'] == 'succeeded' and receipt['step'] == 1
    saved = [json.loads(path.read_text()) for path in (tmp_path / 'run/checkpoints').glob('*/manifest.json')]
    assert len([record for record in saved if ids[0] in record.get('request_ids', [])]) == 1


def test_manual_checkpoint_failure_is_rejected_without_changing_training(tmp_path, monkeypatch):
    import hypergan.single_execution as execution
    from hypergan.run_requests import submit_checkpoint_request, checkpoint_request_status
    config = write_default(tmp_path / 'config', device="cpu")
    train(config, tmp_path / 'full')
    original = execution.write_checkpoint
    def fail_manual(run_dir, trainer, batch, metadata):
        if metadata.get('request_ids'):
            raise OSError('manual checkpoint volume unavailable')
        return original(run_dir, trainer, batch, metadata)
    monkeypatch.setattr(execution, 'write_checkpoint', fail_manual)
    ids = []
    def observer(row):
        if row['event'] == 'train' and row['step'] == 2:
            request = submit_checkpoint_request(tmp_path / 'run', run_id=row['run_id'], attempt_id=row['attempt_id'])
            ids.append(request['request']['request_id'])
    result = train(config, tmp_path / 'run', on_event=observer)
    receipt = checkpoint_request_status(tmp_path / 'run', ids[0])
    assert receipt['status'] == 'rejected' and 'unavailable' in receipt['error']
    assert result['status'] == 'complete' and result['observation_errors'][0]['source'] == 'manual_checkpoint'
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'run')[2])


def test_corrupt_old_preview_metadata_does_not_accumulate_new_orphans(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    stopped = train(config, tmp_path / 'run', stop_after_steps=1, preview_every=1)
    old = Path(stopped['previews'][0]['path']).parent
    (old / 'manifest.json').write_text('{broken')
    result = resume(tmp_path / 'run')
    assert result['status'] == 'complete' and len(result['observation_errors']) == 4
    assert [path for path in (tmp_path / 'run/previews').iterdir() if path.is_dir()] == [old]
