"""Config-first API POC: model files, lowering, round trips, training and reading runs."""

from pathlib import Path

import tomllib

import pytest

import hypergan.api as hg

from hypergan.config import DEFAULT, config_values, fingerprint, load_config, resolve_config

from hypergan.model_file import ModelFileError, lower

from hypergan.toml_writer import dumps

ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = ROOT / 'examples' / 'api' / 'config_first'

@pytest.fixture(autouse=True)
def project_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))

def small(model, steps=3):
    return hg.override(model, {'train.steps': steps})

def test_item_data_is_seeded_resumable_and_rolls_back():
    import torch
    from hypergan.item_data import ItemData
    data = ItemData('toy_project:GridPoints', {'side': 3, 'size': 10})
    rng = torch.Generator().manual_seed(1)
    first = data(4, generator=rng)['real']
    state, rng_state = data.state_dict(), rng.get_state()
    expected = torch.cat([data(4, generator=rng)['real'] for _ in range(3)])   # crosses an epoch boundary
    again = ItemData('toy_project:GridPoints', {'side': 3, 'size': 10})
    again.load_state_dict(state)
    rng.set_state(rng_state)
    assert torch.equal(torch.cat([again(4, generator=rng)['real'] for _ in range(3)]), expected)
    assert first.shape == (4, 2) and data.state_dict()['epoch'] == 2
    other = ItemData('toy_project:GridPoints', {'side': 3, 'size': 11})
    with pytest.raises(ValueError, match='identity'):
        other.load_state_dict(state)

    class Broken:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            if index == 3:
                raise OSError('unreadable')
            return {'real': torch.zeros(2), 'meta': {'id': index, 'pair': [torch.ones(1), 2.0]}}
    import toy_project
    toy_project.Broken = Broken
    broken = ItemData('toy_project:Broken', shuffle=False)
    batch = broken(2, generator=rng)
    assert batch['meta']['id'].tolist() == [0, 1] and batch['meta']['pair'][0].shape == (2, 1)
    before = broken.state_dict()
    with pytest.raises(OSError):
        broken(2, generator=rng)
    assert broken.state_dict() == before

def test_plain_function_metric_and_evaluation_adapters():
    import torch
    from hypergan.plugin_functions import ScalarFunction, SnapshotFunction
    metric = ScalarFunction('toy_project:d_over_g', direction='none')
    assert metric.describe()['kind'] == 'scalar' and 'source_sha256=' in metric.describe()['description']
    assert metric.evaluate(context={}, d=1.0, g=4.0) == 0.25
    evaluation = SnapshotFunction('toy_project:modes_covered', args={'side': 2}, direction='maximize')
    corners = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    batches = iter([{'generated': corners[:1], 'reference': corners[:1]}, {'generated': corners[1:], 'reference': corners[1:]}])
    assert evaluation.evaluate(batches=batches, context={}) == 0.5

def test_train_simple_model_and_read_metrics(tmp_path):
    model = small(hg.load(EXAMPLES / 'simple.toml'))
    run = hg.train(model, tmp_path / 'run')
    assert (run.status, run.step, run.fingerprint) == ('complete', 3, hg.fingerprint(model))
    series = hg.metrics(run)
    assert [step for step, _ in series['loss/d_total']] == [1, 2, 3]
    assert set(hg.metrics(run, ['loss/g_total'])) == {'loss/g_total'}
    fresh = hg.sample(run, count=5, seed=1)
    assert tuple(fresh.data.shape) == (5, 2) and fresh.step == 3
    image = hg.view(fresh, 'toy_project:scatter', run=run)
    assert (image.width, image.height, image.channels) == (64, 64, 1)
    assert image.png().startswith(b'\x89PNG')
    assert not any(p.suffix in ('.py', '.hndl') for p in run.path.rglob('*'))   # no user code copied

def test_item_data_run_stops_and_resumes_at_the_same_position(tmp_path):
    model = small(hg.override(hg.load(EXAMPLES / 'simple.toml'),
                              {'data': {'dataset': 'toy_project:GridPoints', 'args': {'size': 100}}}), steps=4)
    straight = hg.train(model, tmp_path / 'straight')
    hg.train(model, tmp_path / 'split', stop_after_steps=2)
    split = hg.train(model, tmp_path / 'split')          # same model file, same run dir: continue
    assert split.step == 4
    left, right = hg.metrics(straight)['loss/d_total'], hg.metrics(split)['loss/d_total']
    assert [v for s, v in left if s > 2] == pytest.approx([v for s, v in right if s > 2][-2:])

def test_multi_network_model_trains(tmp_path):
    run = hg.train(small(hg.load(EXAMPLES / 'two_critics.toml')), tmp_path / 'run')
    series = hg.metrics(run)
    assert run.step == 3 and 'loss/objectives/reconstruction' in series
    assert run.config['adversarial_terms'][0]['component'] == 'marginal_critic'

@pytest.mark.heavy
def test_extensions_run_metrics_evaluations_previews_and_samplers(tmp_path):
    import demo_3_extensions
    run = demo_3_extensions.main(tmp_path / 'run')
    assert 'd_over_g' in hg.metrics(run)
    results = hg.evaluations(run)
    assert results['nearest'][0]['status'] == 'complete' and results['modes'][0]['step'] == 10
    previews = hg.samples(run)
    assert previews and tuple(previews[0].data.shape) == (16, 2)
    assert hg.view(previews[0], 'scatter', run=run).width == 64
