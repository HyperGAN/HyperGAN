"""Observation never changes numerical state or publishes partial artifacts."""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from tests.hndl_fixtures import fixture_network
from torch import nn

from hypergan.checkpoints import read_checkpoint, trainer_state
from hypergan.config import load_config, write_default
from hypergan.previews import MAX_BYTES, MAX_COUNT, render_preview
from hypergan.training import ReferenceTrainer, resume, train


@pytest.fixture(autouse=True)
def importable_preview_factories(monkeypatch):
    # CPU preview workers reconstruct ordinary importable recipe factories.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))


class StochasticGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = fixture_network('stochastic_features', (4,), (2,), width=2, probability=0.3)
        self.register_buffer('counter', torch.zeros(()), persistent=False)

    def forward(self, x):
        self.counter.add_(1)
        value = self.network(x)
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


@pytest.mark.heavy
def test_preview_schedule_retention_and_resume_preserve_complete_state(tmp_path):
    config = stochastic_config(tmp_path / 'config.toml')
    full = train(config, tmp_path / 'full')
    stopped = train(config, tmp_path / 'observed', preview_every=1, preview_keep=2, stop_after_steps=3)
    assert 1 <= len(stopped['previews']) <= 2
    initial_paths = {record['path'] for record in stopped['previews']}
    initial_sequence = stopped['next_sample_sequence']
    done = resume(tmp_path / 'observed')
    assert done['preview_every'] == 1 and done['preview_keep'] == 2
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'observed')[2])
    assert done['next_sample_sequence'] > initial_sequence
    index = json.loads((tmp_path / 'observed/previews/index.json').read_text())
    steps = [record['step'] for record in index['previews']]
    assert steps == sorted(steps) and 4 <= steps[-1] <= 6
    retained = {record['path'] for record in index['previews']}
    assert all(Path(path).exists() == (path in retained) for path in initial_paths)
    directories = [path for path in (tmp_path / 'observed/previews').iterdir() if path.is_dir()]
    assert len(directories) == len(index['previews']) <= 2
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
    import hypergan.preview_worker as worker
    config = write_default(tmp_path / 'config', device="cpu")
    train(config, tmp_path / 'full')
    def disk_full(*args, **kwargs):
        raise OSError('preview volume full')
    monkeypatch.setattr(worker, 'render_snapshot', disk_full)
    result = train(config, tmp_path / 'observed', preview_every=1)
    assert result['status'] == 'complete'
    assert 1 <= len(result['observation_errors']) <= 5
    assert result['next_sample_sequence'] == len(result['observation_errors']) + 2
    assert all(error['source'] == 'preview' for error in result['observation_errors'])
    assert not (tmp_path / 'observed/previews').exists()
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'observed')[2])


def test_preview_byte_limit_is_enforced_before_publication(tmp_path, monkeypatch):
    import hypergan.previews as previews
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'config', device="cpu")))
    _, batch = trainer.update()
    monkeypatch.setattr(previews, 'MAX_BYTES', 80)
    (tmp_path / 'run').mkdir()
    with pytest.raises(ValueError, match='byte budget'):
        previews.publish_preview(tmp_path / 'run', trainer, batch,
            {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1})
    assert not [path for path in (tmp_path / 'run/previews').iterdir() if path.is_dir()]


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
    assert all(receipt['status'] == 'succeeded' and 2 <= receipt['step'] <= 5 for receipt in receipts)
    assert receipts[0]['checkpoint_path'] == receipts[1]['checkpoint_path']
    _, metadata, state = read_checkpoint(tmp_path / 'run', receipts[0]['checkpoint_path'])
    assert set(metadata['request_ids']) == set(requests) and state['step'] == receipts[0]['step']
    assert result['status'] == 'complete'


def test_manual_request_is_never_serviced_after_partial_update(tmp_path, monkeypatch):
    from hypergan.run_requests import submit_checkpoint_request, checkpoint_request_status
    config = write_default(tmp_path / 'config', device="cpu")
    requests = []
    from hypergan.training import DeviceAdam
    original = DeviceAdam.step
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
    monkeypatch.setattr(DeviceAdam, 'step', fail_generator)
    with pytest.raises(KeyboardInterrupt):
        train(config, tmp_path / 'run')
    assert read_checkpoint(tmp_path / 'run')[2]['step'] == 0
    assert checkpoint_request_status(tmp_path / 'run', requests[0])['status'] == 'pending'
    monkeypatch.setattr(DeviceAdam, 'step', original)
    resume(tmp_path / 'run')
    receipt = checkpoint_request_status(tmp_path / 'run', requests[0])
    assert receipt['status'] == 'rejected' and 'attempt' in receipt['error']


class MutatingConditionGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = fixture_network('condition_projection',
            {'x': (4,), 'condition': (2,)}, (2,))

    def forward(self, x, condition):
        condition.add_(2)
        return self.network(x=x, condition=condition)


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


@pytest.mark.heavy
def test_resume_refreshes_a_stale_default_preview_bound(tmp_path):
    """A manifest that stored an old default must not pin the run to it."""
    from hypergan.previews import DEFAULT_KEEP
    config = write_default(tmp_path / 'config', device='cpu')
    run = tmp_path / 'run'
    stopped = train(config, run, preview_every=1, preview_keep=2, stop_after_steps=1)
    assert stopped['preview_keep'] == 2 and stopped['preview_keep_source'] == 'explicit'
    manifest = json.loads((run / 'manifest.json').read_text())

    # Rewritten in the shape an older release left behind: a bound of 20 that was
    # only ever the default of the day, with no marker saying so.
    manifest['preview_keep'] = 20
    manifest.pop('preview_keep_source', None)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    refreshed = resume(run, preview_every=1, stop_after_steps=1)
    assert refreshed['preview_keep'] == DEFAULT_KEEP == 128
    assert refreshed['preview_keep_source'] == 'default'

    # An explicit bound on this attempt is recorded as explicit and inherited.
    bounded = resume(run, preview_every=1, preview_keep=3, stop_after_steps=1)
    assert bounded['preview_keep'] == 3 and bounded['preview_keep_source'] == 'explicit'
    inherited = resume(run, preview_every=1, stop_after_steps=1)
    assert inherited['preview_keep'] == 3 and inherited['preview_keep_source'] == 'explicit'
    index = json.loads((run / 'previews/index.json').read_text())
    assert index['keep'] == 3 and index['retention'] == 'thinned'
    sequences = [item['identity']['sample_sequence'] for item in index['previews']]
    assert len(sequences) <= 3 and sequences[0] == 1


@pytest.mark.heavy
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
    assert len(result['previews']) == 1 and 2 <= result['previews'][0]['step'] <= 5


def test_failed_index_write_does_not_accumulate_published_orphans(tmp_path, monkeypatch):
    import hypergan.previews as previews
    original = previews.atomic_json
    def fail_index(path, value):
        if Path(path).name == 'index.json':
            raise OSError('index write failed')
        return original(path, value)
    monkeypatch.setattr(previews, 'atomic_json', fail_index)
    trainer = ReferenceTrainer(load_config(write_default(tmp_path / 'config', device="cpu")))
    _, batch = trainer.update()
    (tmp_path / 'run').mkdir()
    with pytest.raises(OSError, match='index write failed'):
        previews.publish_preview(tmp_path / 'run', trainer, batch,
            {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1})
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
    assert receipt['status'] == 'succeeded' and 1 <= receipt['step'] <= 5
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


@pytest.mark.heavy
def test_corrupt_old_preview_metadata_does_not_accumulate_new_orphans(tmp_path):
    config = write_default(tmp_path / 'config', device="cpu")
    stopped = train(config, tmp_path / 'run', stop_after_steps=1, preview_every=1)
    previews = tmp_path / 'run/previews'
    generations = lambda: sorted(path for path in previews.iterdir() if path.is_dir())
    old = Path(stopped['previews'][0]['path']).parent
    (old / 'manifest.json').write_text('{broken')
    # The published index already names that generation, so its manifest is not
    # reread and the corruption no longer stops the run from publishing.
    result = resume(tmp_path / 'run', stop_after_steps=1)
    assert not result['observation_errors'] and old in generations()
    assert old in {Path(record['path']).parent for record in result['previews']}
    # Without a readable index the scan falls back to rereading every manifest,
    # so each publication fails - and removes the generation it just published.
    (previews / 'index.json').write_text('{broken')
    intact = generations()
    result = resume(tmp_path / 'run')
    assert result['status'] == 'complete' and 1 <= len(result['observation_errors']) <= 4
    assert generations() == intact
