"""Starter recipes keep the requested template and need no path into the installed package."""
from pathlib import Path
import tomllib

import pytest

import hypergan
import hypergan.api as hg
from hypergan.recipe_generator import init_project


@pytest.mark.parametrize('template', ['dcgan', 'resnet', 'tiny-transformer'])
def test_explicit_image_template_without_data_is_an_error_not_the_toy(tmp_path, template):
    with pytest.raises(ValueError, match='needs image data: pass --dataset DIR') as error:
        init_project(tmp_path / 'p', template=template, device='cpu')
    assert 'gaussian-grid' in str(error.value)
    assert not (tmp_path / 'p' / 'configs' / 'gaussian-grid.toml').exists()


def test_explicit_template_with_resolution_or_data_is_kept(tmp_path):
    info = init_project(tmp_path / 'p', template='dcgan', resolution=32, device='cpu')
    assert info['template'] == 'dcgan'
    assert Path(info['recipe_path']).name == 'dcgan.toml'
    assert hg.load(info['recipe_path']).spec['networks']['generator']['output_shape'] == ['B', 3, 32, 32]


def test_no_template_picks_the_toy_or_dcgan_from_the_inputs(tmp_path):
    assert init_project(tmp_path / 'toy', device='cpu')['template'] == 'gaussian-grid'
    assert init_project(tmp_path / 'img', resolution=32, device='cpu')['template'] == 'dcgan'


def test_api_new_project_passes_the_template_through(tmp_path):
    with pytest.raises(ValueError, match='needs image data'):
        hg.new(project=tmp_path / 'p', template='dcgan', device='cpu')


def _assert_self_contained(path):
    text = Path(path).read_text(encoding='utf-8')
    assert 'hndl' not in text and 'site-packages' not in text and str(Path(hg.__file__).parent) not in text
    spec = hg.load(path).spec
    assert all('source' in network for network in spec['networks'].values())


def test_gaussian_grid_starters_inline_the_reference_network(tmp_path):
    saved = hg.save(hg.packaged('gaussian-grid'), tmp_path / 'elsewhere' / 'a1.toml')
    _assert_self_contained(saved)
    info = init_project(tmp_path / 'proj', device='cpu')
    _assert_self_contained(info['recipe_path'])
    assert not (tmp_path / 'proj' / 'networks').exists()
    assert hg.fingerprint(hg.load(saved)) == hg.fingerprint(hg.packaged('gaussian-grid'))


def test_packaged_gaussian_grid_inlines_reference_hndl_byte_for_byte():
    package = Path(hypergan.__file__).parent
    reference = (package / 'networks' / 'reference.hndl').read_text(encoding='utf-8')
    networks = tomllib.loads((package / 'models' / 'gaussian-grid.toml').read_text(encoding='utf-8'))['networks']
    assert {name: spec['source'] for name, spec in networks.items()} == {
        'generator': reference, 'discriminator': reference}


def test_gaussian_grid_starter_is_the_qualified_reference_when_prepared(tmp_path):
    saved = hg.save(hg.override(hg.packaged('gaussian-grid'), {'train.device': 'cpu'}), tmp_path / 'a1.toml')
    prepared = hg.prepare(hg.load(saved), tmp_path / 'run')
    assert Path(prepared.config['_source_path']) == Path(saved).resolve()
    assert prepared.config['qualification']['recipe_match'] is True
    assert not any('unqualified' in warning for warning in prepared.config['warnings'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('steps', [None, 20])
def test_gaussian_grid_starter_flow_is_the_reference_recipe(tmp_path, device, steps):
    # README flow: `hypergan new --project p --template gaussian-grid [--device D]`,
    # then `hypergan train p/configs/gaussian-grid.toml [--steps 20]`.
    info = init_project(tmp_path / 'p', template='gaussian-grid', device=device)
    prepared = hg.prepare(hg.load(info['recipe_path']), tmp_path / 'run', steps)
    config = prepared.config
    assert (config['training']['device'], config['training']['steps']) == (device, steps or 5)
    assert config['qualification']['recipe_match'] is True
    assert not any('unqualified' in warning or 'trusted Python' in warning for warning in config['warnings'])
    wider = hg.override(hg.load(info['recipe_path']), {'train.batch_size': 32})
    assert hg.resolve(wider)['qualification']['recipe_match'] is False
