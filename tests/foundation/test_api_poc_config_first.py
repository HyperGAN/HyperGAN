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


# -- files ----------------------------------------------------------------------

@pytest.mark.parametrize('path', sorted((ROOT / 'examples').glob('*.toml')) + sorted(EXAMPLES.glob('*.toml'))
                         + sorted((ROOT / 'src' / 'hypergan' / 'models').glob('*.toml')), ids=lambda p: p.name)
def test_toml_writer_round_trips_every_example(path):
    raw = tomllib.loads(path.read_text())
    for depth in (0, 3):
        assert tomllib.loads(dumps(raw, table_depth=depth)) == raw


def test_toml_writer_escapes_and_omits_none():
    doc = {'s': 'quote " and """ and \\ and\ttab', 'm': 'line\nends with "', 'n': None, 'weird key': {'x': -0.0},
           'rows': [{'a': 1, 'inner': {'b': [1, 2.5, True]}}, {'c': [{'d': 'e'}]}]}
    back = tomllib.loads(dumps(doc))
    assert back == {k: v for k, v in doc.items() if v is not None}
    with pytest.raises(ValueError):
        dumps({'x': float('nan')})


def test_model_file_lowers_roles_and_the_loss_list():
    recipe = hg.lower(hg.load(EXAMPLES / 'two_critics.toml'))
    components = recipe['components']
    assert set(components) == {'encoder', 'generator', 'discriminator', 'marginal_critic'}
    assert components['generator']['inputs'] == {'x': 'latent', 'condition': 'components.encoder'}
    assert components['discriminator']['inputs'] == {'x': 'candidate', 'condition': 'batch.condition'}
    assert 'linear()' in components['marginal_critic']['args']['source']  # read from critic.hndl
    assert recipe['adversarial'] == {'weight': 1.0}
    assert recipe['gradient_penalty'] == {'coeff': 1.0, 'kappa': 1.0, 'lazy_k': 1}
    assert recipe['adversarial_terms'] == [{'id': 'marginal_critic', 'component': 'marginal_critic', 'weight': 0.5,
                                            'penalty': True, 'penalty_coeff': 0.5, 'real': 'batch.real', 'fake': 'generated'}]
    assert recipe['objectives'] == [{'id': 'reconstruction', 'factory': 'l1', 'weight': 1.0, 'args': {},
                                     'inputs': {'input': 'generated', 'target': 'batch.real'}}]
    assert recipe['prior_regularizer'] == {'weight': 1.0}
    assert recipe['data'] == {'factory': 'hypergan.item_data:ItemData',
                              'args': {'dataset': 'toy_project:PairedVectors', 'args': {'size': 512, 'dims': 2}}}
    resolve_config(recipe)


def test_packaged_model_is_the_reference_recipe():
    model = hg.packaged('gaussian-grid')
    assert hg.fingerprint(model) == fingerprint(resolve_config({}))
    assert hg.resolve(model)['qualification']['recipe_match'] is True
    with pytest.raises(ValueError, match='gaussian-grid'):
        hg.packaged('nope')


def test_save_and_load_keep_the_fingerprint_and_reference_hndl_in_place(tmp_path):
    original = hg.load(EXAMPLES / 'two_critics.toml')
    saved = hg.save(original, tmp_path / 'elsewhere' / 'model.toml')
    reloaded = hg.load(saved)
    assert hg.fingerprint(reloaded) == hg.fingerprint(original)
    assert fingerprint(load_config(saved)) == hg.fingerprint(original)   # the CLI's loader
    assert not list(saved.parent.glob('*.hndl'))                         # referenced, not copied
    hndl = reloaded.spec['networks']['marginal_critic']['hndl']
    assert (saved.parent / hndl).resolve() == (EXAMPLES / 'critic.hndl').resolve()
    import demo_4_round_trip
    built = hg.save(demo_4_round_trip.python_model(), tmp_path / 'built.toml')
    assert hg.fingerprint(hg.load(built)) == hg.fingerprint(original)


def test_override_addresses_losses_by_id_and_merges_sections():
    model = hg.load(EXAMPLES / 'two_critics.toml')
    changed = hg.override(model, {'losses.reconstruction.weight': 0.25, 'losses.marginal_critic.penalty': 0.0},
                          train={'steps': 3})
    recipe = hg.lower(changed)
    assert recipe['objectives'][0]['weight'] == 0.25
    assert recipe['adversarial_terms'][0]['penalty'] is False and 'penalty_coeff' not in recipe['adversarial_terms'][0]
    assert recipe['training']['steps'] == 3 and recipe['training']['batch_size'] == 16
    assert hg.fingerprint(changed) != hg.fingerprint(model)
    assert model.spec['losses'][2]['weight'] == 1.0                    # originals are not mutated


@pytest.mark.parametrize('change, message', [
    ({'networks.encoder.role': 'teacher'}, 'networks.encoder.role'),
    ({'networks.encoder.role': 'generator'}, 'Exactly one network'),
    ({'losses.0.critic': 'encoder'}, 'critic must name'),
    ({'networks.decoder.inputs.condition': 'encodr'}, "'encodr' is neither a network"),
    ({'losses.0.real': 'batch.condition'}, 'first adversarial loss'),
    ({'losses.reconstruction.id': None}, 'id is required'),
    ({'networks.encoder.colour': 'red'}, 'Unknown networks.encoder field'),
    ({'train.learning_rate': 1.0}, 'Unknown train field'),
    ({'penalty.coeff': 2.0}, 'penalty.coeff'),
    ({'networks.marginal_critic.role': 'auxiliary'}, 'critic must name'),
])
def test_errors_name_the_field(change, message):
    model = hg.load(EXAMPLES / 'two_critics.toml')
    spec = hg.override(model, change).spec
    if change == {'losses.reconstruction.id': None}:
        del spec['losses'][2]['id']
    with pytest.raises(ModelFileError, match=message.replace('(', r'\(')):
        lower(spec, EXAMPLES)


def test_unused_critic_and_missing_prior_loss_are_reported():
    model = hg.load(EXAMPLES / 'two_critics.toml')
    spec = dict(model.spec, losses=[loss for loss in model.spec['losses'] if loss.get('critic') != 'marginal_critic'])
    with pytest.raises(ModelFileError, match='marginal_critic are in no adversarial loss'):
        lower(spec, EXAMPLES)
    no_prior = hg.from_dict(dict(model.spec, losses=[l for l in model.spec['losses'] if l['type'] != 'prior']), base=EXAMPLES)
    assert hg.lower(no_prior)['prior_regularizer'] == {'weight': 0.0}
    assert any('No type = "prior" loss' in warning for warning in hg.validate(no_prior))


def test_load_config_accepts_model_files_and_mappings_without_changing_recipes():
    paired = ROOT / 'examples' / 'paired-linear.toml'
    assert fingerprint(load_config(paired)) == fingerprint(resolve_config(tomllib.loads(paired.read_text())))
    resolved = load_config(EXAMPLES / 'simple.toml')
    assert fingerprint(load_config(resolved)) == fingerprint(resolved)               # resolved mapping
    spec = hg.load(EXAMPLES / 'simple.toml').spec
    with pytest.raises(ModelFileError, match='no file location'):     # a mapping has no base directory
        load_config(spec)
    spec['networks']['critic'] = dict(spec['networks']['critic'], hndl=str(EXAMPLES / 'critic.hndl'))
    assert fingerprint(load_config(spec)) == fingerprint(resolved)                  # model-file mapping
    assert fingerprint(load_config({})) == fingerprint(resolve_config({}))


def test_samplers_are_recorded_but_not_fingerprinted():
    model = hg.load(EXAMPLES / 'extensions.toml')
    without = hg.from_dict({k: v for k, v in model.spec.items() if k != 'samplers'}, base=EXAMPLES)
    assert hg.fingerprint(model) == hg.fingerprint(without)
    resolved = hg.resolve(model)
    assert config_values(resolved)['samplers'] == {'scatter': {'fn': 'toy_project:scatter', 'args': {'size': 64}}}
    assert 'samplers' not in config_values(hg.resolve(without))
    with pytest.raises(ValueError, match='samplers.bad.fn'):
        resolve_config(dict(hg.lower(model), samplers={'bad': {'fn': 'no_colon'}}))


def test_metrics_and_evaluations_lower_to_plugin_adapters():
    custom = hg.lower(hg.load(EXAMPLES / 'extensions.toml'))['metrics']['custom']
    assert custom['d_over_g'] == {'mode': 'scalar', 'inputs': {'d': 'update.d_loss', 'g': 'update.g_loss'},
                                  'every_steps': 10, 'factory': 'hypergan.plugin_functions:ScalarFunction',
                                  'args': {'fn': 'toy_project:d_over_g', 'direction': 'none'}}
    modes = custom['modes']
    assert modes['trigger'] == 'interval' and modes['every_steps'] == 10
    assert modes['inputs'] == {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'}
    assert modes['evaluation']['data']['args']['args']['split'] == 'holdout'
    assert custom['nearest']['trigger'] == 'manual'


# -- user code adapters ----------------------------------------------------------

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


# -- training and reading runs ------------------------------------------------------

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
