"""Per-network API proof of concept: declarations, lowering, round trip, training, reading."""
import glob
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'examples' / 'api' / 'per-network'
sys.path.insert(0, str(EXAMPLES))

from hypergan import api_per_network as hg  # noqa: E402
from hypergan.api_per_network import toml_io  # noqa: E402
from hypergan.config import fingerprint, load_config  # noqa: E402
from toy_project import recipes  # noqa: E402
from toy_project.nets import Encoder, PointGenerator  # noqa: E402

CRITIC = recipes.CRITIC


def _pair(**overrides):
    model = {
        'generator': hg.generator(hg.net(PointGenerator), inputs={'z': hg.LATENT},
                                  losses=[hg.adversarial('discriminator')]),
        'discriminator': hg.critic(hg.hndl(CRITIC, input_shape=['B', 2], output_shape=['B', 1]),
                                   inputs={'x': hg.CANDIDATE}),
    }
    model.update(overrides)
    return model


def _recipe(model, **options):
    options.setdefault('data', hg.data('gaussian_grid', side=3))
    options.setdefault('prior', hg.particles(z_dim=4, count=32))
    options.setdefault('training', hg.training(steps=3, batch_size=8, device='cpu'))
    return hg.recipe(model, **options)


# ------------------------------------------------------------ declarations and lowering

def test_user_classes_are_referenced_by_import_path_never_copied():
    assert hg.net(PointGenerator, width=8) == hg.Network('toy_project.nets:PointGenerator', {'width': 8})

    class Local:
        pass
    with pytest.raises(ValueError, match='module level'):
        hg.net(Local)
    Local.__module__, Local.__qualname__ = '__main__', 'Local'
    with pytest.raises(ValueError, match='importable module'):
        hg.net(Local)


def test_simple_recipe_lowers_to_the_existing_config():
    config = hg.lower(recipes.simple())
    assert config['components']['generator']['factory'] == 'toy_project.nets:PointGenerator'
    assert config['components']['discriminator']['inputs'] == {'x': 'candidate'}
    assert 'objectives' not in config and 'adversarial_terms' not in config
    resolved = hg.validate(recipes.simple())
    assert resolved['data'] == {'factory': 'gaussian_grid', 'args': {'side': 5, 'noise': 0.05}}


def test_multi_network_losses_and_critics_lower_per_network():
    config = hg.lower(recipes.multi_network())
    assert [(t['id'], t['inputs']) for t in config['objectives']] == [
        ('encoder.code_norm', {'code': 'components.encoder'}),
        ('generator.reconstruction', {'input': 'generated', 'target': 'batch.real'})]
    assert config['adversarial_terms'] == [{'id': 'marginal.main', 'component': 'marginal', 'real': 'batch.real',
                                            'fake': 'generated', 'weight': 0.5, 'penalty': True,
                                            'penalty_coeff': 0.5}]
    assert config['optimizer'] == {'lr': 6e-4, 'betas': [0.0, 0.999], 'd_lr_mult': 1.5, 'prior_lr_mult': 10.0}
    assert config['gradient_penalty'] == {'coeff': 1.0}
    assert Path(config['components']['marginal']['args']['file']).is_absolute()
    text = hg.explain(recipes.multi_network())
    assert 'gradient from: fooling discriminator, fooling marginal, loss encoder.code_norm, loss generator.reconstruction' in text


def test_a_judged_network_must_declare_the_critics_it_fools():
    model = _pair(generator=hg.generator(hg.net(PointGenerator), inputs={'z': hg.LATENT}))
    with pytest.raises(ValueError, match=r"add adversarial\('discriminator'\)"):
        hg.lower(_recipe(model))
    model = _pair(generator=hg.generator(hg.net(PointGenerator), inputs={'z': hg.LATENT},
                                         losses=[hg.adversarial('discriminator'), hg.adversarial('other')]))
    with pytest.raises(ValueError, match='names no declared critic'):
        hg.lower(_recipe(model))


def test_a_loss_must_reach_the_network_it_is_attached_to():
    model = _pair(
        encoder=hg.encoder(hg.net(Encoder), inputs={'condition': 'batch.condition'},
                           losses=[hg.mse('generated', 'batch.real')]))
    with pytest.raises(ValueError, match='do not depend on encoder'):
        hg.lower(_recipe(model))


def test_engine_limits_are_explicit_errors():
    with pytest.raises(ValueError, match="networks\\['generator'\\]"):
        hg.lower(_recipe({'g': _pair()['generator'], 'discriminator': _pair()['discriminator']}))
    model = _pair(encoder=hg.encoder(hg.net(Encoder), inputs={'condition': 'batch.condition'},
                                     optimizer=hg.adam(lr=1e-3)))
    with pytest.raises(ValueError, match='per-network optimizers'):
        hg.lower(_recipe(model))
    model = _pair(
        generator=hg.generator(hg.net(PointGenerator), inputs={'z': hg.LATENT},
                               losses=[hg.adversarial('discriminator'), hg.adversarial('second')]),
        second=hg.critic(hg.hndl(CRITIC, input_shape=['B', 2], output_shape=['B', 1]), inputs={'x': hg.CANDIDATE},
                         penalty=hg.k3p(coeff=1.0, kappa=2.0)))
    with pytest.raises(ValueError, match='only coeff may differ'):
        hg.lower(_recipe(model))


def test_frozen_and_shared_roles_lower_to_trainable_false_and_reuse():
    from toy_project.nets import FixedFeatures
    model = _pair(
        features=hg.frozen(hg.net(FixedFeatures), inputs={'x': 'batch.real'}),
        discriminator=hg.critic(hg.hndl('concat(x, features)\nlinear(8)\nleaky_relu(0.2)\nlinear()',
                                        input_shape={'x': ['B', 2], 'features': ['B', 4]}, output_shape=['B', 1]),
                                inputs={'x': hg.CANDIDATE, 'features': 'features'}),
        again=hg.shared('generator', inputs={'z': hg.LATENT}, losses=[hg.l1('again', 'batch.real', id='again')]))
    config = hg.lower(_recipe(model))
    assert config['components']['features']['trainable'] is False
    assert config['components']['again'] == {'reuse': 'generator', 'inputs': {'z': 'latent'}}
    assert config['objectives'][0]['id'] == 'again.again'
    hg.validate(_recipe(model))


# ------------------------------------------------------------ config files

@pytest.mark.parametrize('build', [recipes.simple, recipes.multi_network, recipes.extensions])
def test_save_and_load_keep_the_fingerprint(tmp_path, build):
    recipe = build()
    path = hg.save(recipe, tmp_path / 'nested' / 'model.toml')
    assert load_config(path) is not None
    assert hg.fingerprint(path) == hg.fingerprint(recipe) == hg.fingerprint(hg.load(path))
    assert hg.lower(hg.load(path)) == hg.lower(recipe)


def test_saved_file_groups_losses_under_their_network_and_keeps_hndl_files_as_references(tmp_path):
    path = hg.save(recipes.multi_network(), EXAMPLES / 'toy_project' / '_saved_for_test.toml')
    try:
        text = path.read_text()
        assert 'file = "marginal_critic.hndl"' in text
        assert text.index('[components.encoder]') < text.index('id = "encoder.code_norm"') \
            < text.index('[components.generator]') < text.index('id = "generator.reconstruction"') \
            < text.index('[components.discriminator]') < text.index('[components.marginal]') \
            < text.index('id = "marginal.main"')
    finally:
        path.unlink()


@pytest.mark.parametrize('path', sorted(glob.glob(str(ROOT / 'examples' / '*.toml'))), ids=lambda p: Path(p).name)
def test_every_existing_example_config_lifts_to_declarations_with_the_same_fingerprint(tmp_path, path):
    recipe = hg.load(path)
    assert hg.fingerprint(recipe) == fingerprint(load_config(path))
    assert hg.fingerprint(hg.save(recipe, tmp_path / 'again.toml')) == fingerprint(load_config(path))


def test_load_config_accepts_a_raw_mapping_like_its_file(tmp_path):
    raw = hg.lower(recipes.simple())
    path = tmp_path / 'simple.toml'
    path.write_text(toml_io.dumps(raw))
    assert fingerprint(load_config(raw)) == fingerprint(load_config(path))
    assert tomllib.loads(toml_io.dumps(raw)) == raw


def test_toml_writer_round_trips_awkward_values():
    raw = {'name': 'a "quoted" \\ name', 'x': {'loss/total': {'enabled': False}, 'k': [1, 2.5, 1e-05, True],
                                                  'src': "line one\nit's 'quoted'\n", 'tail': "ends with '\n'",
                                                  'empty': {}, 'rows': [{'a': 1}, {'b': 'two'}]}}
    assert tomllib.loads(toml_io.dumps(raw)) == raw


# ------------------------------------------------------------ adapters

def test_item_data_owns_order_and_resume_position():
    import torch
    from hypergan.api_per_network.adapters import ItemData
    source = ItemData('toy_project.data:RingPoints', {'count': 10})
    generator = torch.Generator().manual_seed(0)
    first = [source.indices(4, generator) for _ in range(2)]
    state, stream = source.state_dict(), generator.get_state()
    after = [source.indices(4, generator) for _ in range(3)]
    source.load_state_dict(state)
    generator.set_state(stream)
    assert [source.indices(4, generator) for _ in range(3)] == after
    assert sorted(first[0] + first[1] + after[0][:2]) == list(range(10))  # one full epoch, no repeats
    batch = source(4, generator=generator)
    assert set(batch) == {'real'} and batch['real'].shape == (4, 2)
    paired = ItemData('toy_project.data:PairedPoints', {'count': 8})(3, generator=generator)
    assert set(paired) == {'condition', 'real'} and paired['real'].shape == (3, 2)
    identity = source.resume_identity()
    assert identity['dataset'] == 'toy_project.data:RingPoints' and len(identity['source_sha256']) == 64


def test_function_adapters_call_plain_user_functions():
    import torch
    from hypergan.api_per_network.adapters import FunctionEvaluation, FunctionMetric, function_loss
    metric = FunctionMetric('toy_project.observe:g_over_d', label='ratio')
    assert metric.evaluate(context={}, g_loss=3.0, d_loss=2.0) == 1.5
    assert 'source sha256' in metric.describe()['description']
    evaluation = FunctionEvaluation('toy_project.observe:radius_gap')
    batches = [{'generated': torch.ones(4, 2), 'reference': torch.zeros(4, 2)}]
    assert evaluation.evaluate(batches=iter(batches), context={}) == pytest.approx(2 ** 0.5)
    loss = function_loss('toy_project.observe:code_norm')
    assert loss.resume_stateless is True and float(loss(code=torch.ones(2, 2))) == 1.0


# ------------------------------------------------------------ training and reading runs

def test_train_simple_and_read_metrics_samples_and_previews(tmp_path):
    from toy_project.observe import scatter
    recipe = recipes.simple(steps=4)
    run = hg.train(recipe, tmp_path / 'run', preview_every=2)
    assert (run.status, run.steps, run.config_sha256) == ('complete', 4, hg.fingerprint(recipe))
    metrics = hg.metrics(run)
    assert [step for step, _ in metrics['loss/d_total']] == [1, 2, 3, 4]
    assert hg.catalog(run)['loss/d_total']['label'] == 'Discriminator total'
    view = hg.samples(run, 32, seed=0, sampler=hg.sampler('scatter', scatter))
    assert view['count'] == 32 and view['step'] == 4
    previews = hg.previews(run)
    assert previews and previews[0].shape[1] == 2


def test_train_multi_network_publishes_per_network_loss_ids(tmp_path):
    run = hg.train(recipes.multi_network(steps=3), tmp_path / 'run')
    metrics = hg.metrics(run)
    assert len(metrics['loss/objectives/generator.reconstruction']) == 3
    assert len(metrics['loss/objectives/encoder.code_norm']) == 3


def test_training_from_the_saved_file_is_the_same_run_identity(tmp_path):
    recipe = recipes.simple(steps=2)
    path = hg.save(recipe, tmp_path / 'model.toml')
    in_memory = hg.train(recipe, tmp_path / 'a')
    from_file = hg.train(path, tmp_path / 'b')
    assert in_memory.config_sha256 == from_file.config_sha256
    assert hg.metrics(in_memory)['loss/g_total'] == hg.metrics(from_file)['loss/g_total']


def test_item_data_resume_matches_an_uninterrupted_run(tmp_path):
    recipe = recipes.extensions(steps=6)
    recipe = hg.recipe(recipe.networks, data=recipe.data, prior=recipe.prior, training=recipe.training)
    whole = hg.train(recipe, tmp_path / 'whole')
    part = hg.train(recipe, tmp_path / 'part', stop_after_steps=3, checkpoint_every=3)
    assert part.steps == 3
    resumed = hg.resume(part.path)
    assert resumed.steps == 6
    last = lambda run: hg.metrics(run)['loss/g_total'][-1]  # noqa: E731
    assert last(resumed) == last(whole)


def test_manual_custom_evaluation_runs_through_the_api(tmp_path):
    from toy_project.data import RingPoints
    from toy_project.observe import radius_gap
    base = recipes.simple(steps=2)
    recipe = hg.recipe(base.networks, data=base.data, prior=base.prior, training=base.training, observe=[
        hg.evaluation('radius_gap', radius_gap, data=hg.items(RingPoints, count=64, seed=1),
                      samples=32, batch_size=16, device='cpu')])
    assert hg.lower(recipe)['metrics']['custom']['radius_gap']['trigger'] == 'manual'
    run = hg.train(recipe, tmp_path / 'run')
    receipt = hg.evaluate(run, 'radius_gap')
    assert receipt['status'] == 'complete'
    rows = hg.evaluations(run)['radius_gap']
    assert rows[0]['step'] == 2 and rows[0]['value'] >= 0
