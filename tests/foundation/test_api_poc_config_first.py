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


def test_save_on_another_volume_preserves_absolute_hndl_reference(tmp_path, monkeypatch):
    import os
    original = hg.load(EXAMPLES / 'two_critics.toml')
    def different_drives(paths):
        raise ValueError("Paths don't have the same drive")
    with monkeypatch.context() as patch:
        patch.setattr(os.path, 'commonpath', different_drives)
        saved = hg.save(original, tmp_path / 'elsewhere' / 'model.toml')
    reloaded = hg.load(saved)
    source = reloaded.spec['networks']['marginal_critic']['hndl']
    assert Path(source).is_absolute()
    assert Path(source) == (EXAMPLES / 'critic.hndl').resolve()
    assert hg.fingerprint(reloaded) == hg.fingerprint(original)
    assert fingerprint(load_config(saved)) == hg.fingerprint(original)
    assert not list(saved.parent.glob('*.hndl'))


def test_legacy_file_references_survive_external_cwd_override_and_save(tmp_path, monkeypatch):
    import subprocess
    import sys
    project = tmp_path / 'project'
    project.mkdir()
    model_example = hg.load(ROOT / 'examples' / 'cifar-pretrained-sagan.toml')
    raw = hg.lower(model_example)
    raw.pop('format', None)
    raw.pop('name_map', None)
    critic = ROOT / 'examples' / model_example.spec['networks']['discriminator']['hndl']
    (project / 'critic.hndl').write_bytes(critic.read_bytes())
    raw['components']['discriminator']['args'].pop('source', None)
    raw['components']['discriminator']['args']['file'] = 'critic.hndl'
    attention = ROOT / 'src' / 'hypergan' / 'networks' / 'image_attention.hndl'
    net_files_map = {
        'generator': {
            'image_generator': '../src/hypergan/networks/image_generator.hndl',
            'image_attention': '../src/hypergan/networks/image_attention.hndl',
        },
        'encoder': {
            'image_encoder': '../src/hypergan/networks/image_encoder.hndl',
        },
    }
    for comp_name, files in net_files_map.items():
        raw['components'][comp_name]['args'].pop('networks', None)
        raw['components'][comp_name]['args']['network_files'] = dict(files)
    for component in raw['components'].values():
        files = component.get('args', {}).get('network_files', {})
        for name, path in files.items():
            source = ROOT / 'examples' / path
            (project / source.name).write_bytes(source.read_bytes())
            files[name] = source.name
    config = project / 'config.toml'
    config.write_text(dumps(raw))
    expected = load_config(config)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    model = hg.load(config)
    assert model.spec == raw
    lowered = hg.lower(model)
    assert 'network_files' not in lowered['components']['generator']['args']
    assert lowered['components']['generator']['args']['networks']['image_attention'] == attention.read_text()
    assert lowered['components']['discriminator']['args']['source'] == critic.read_text()
    assert lowered['optimizer'] == raw['optimizer']
    assert hg.resolve(model) == expected
    assert hg.validate(model) == expected['warnings']
    assert hg.fingerprint(model) == fingerprint(expected)
    overridden = hg.override(model, {'training.steps': 10})
    assert overridden.base == model.base and hg.resolve(overridden)['training']['steps'] == 10

    saved = hg.save(model, elsewhere / 'saved.toml')
    assert hg.resolve(hg.load(saved)) == expected
    assert fingerprint(load_config(saved)) == fingerprint(expected)
    assert not list(elsewhere.glob('*.hndl'))
    assert 'file' in tomllib.loads(saved.read_text())['components']['discriminator']['args']
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import hypergan.api as hg; "
            "assert hg.fingerprint(hg.load(sys.argv[2])) == sys.argv[3]; "
            "assert not {'torch','hndl','particlegan'} & sys.modules.keys()")
    subprocess.run([sys.executable, '-I', '-c', code, str(Path(hg.__file__).parents[1]),
                    str(config), fingerprint(expected)], cwd=elsewhere, check=True)


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
    assert fingerprint(load_config(paired)) == fingerprint(resolve_config(lower(tomllib.loads(paired.read_text()))))
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


# -- training and reading runs ------------------------------------------------------
