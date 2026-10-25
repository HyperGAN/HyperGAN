"""The file a user reads is the model that trains: no silent substitutions while loading."""
from pathlib import Path
import re
import tomllib

import pytest

import hypergan.api as hg
from hypergan.config import load_config
from hypergan.model_file import NETWORK_FIELDS, ModelFileError
from hypergan.workspace import default_run_dir, interpolate_model_paths, load_workspace

ROOT = Path(__file__).resolve().parents[2]


def _model_with_hndl(tmp_path, hndl):
    text = hg.save(hg.packaged('gaussian-grid'), tmp_path / 'm.toml').read_text(encoding='utf-8')
    spec = tomllib.loads(text)
    spec['networks']['generator'].pop('source')
    spec['networks']['generator']['hndl'] = hndl
    model = tmp_path / 'm.toml'
    hg.save(hg.from_dict(spec, base=tmp_path), model)
    return model


def test_missing_relative_reference_hndl_is_an_error_in_every_loader(tmp_path):
    model = _model_with_hndl(tmp_path, 'nets/reference.hndl')
    with pytest.raises(ValueError, match='cannot read .*nets/reference.hndl'):
        load_config(model)
    with pytest.raises(ValueError, match='cannot read .*nets/reference.hndl'):
        hg.validate(hg.load(model))


def test_engine_recipe_missing_network_file_names_the_field(tmp_path):
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text('[components.generator]\nfactory = "hndl"\ninputs = { x = "latent" }\n'
                      '[components.generator.args]\nfile = "reference.hndl"\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'components\.generator\.args\.file: cannot read reference\.hndl'):
        load_config(recipe)


@pytest.mark.parametrize('text', [
    'format = "hypergan-model/1"\nname = "x"\n',
    'format = "hypergan-model/1"\nname = "x"\n[training]\ndevice = "cpu"\n',
    '[training]\ndevice = "cpu"\nsteps = 3\n',
], ids=['format-only', 'internal-training-table', 'no-format-internal-table'])
def test_file_without_networks_is_rejected_not_the_packaged_default(tmp_path, text):
    path = tmp_path / 'model.toml'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ModelFileError, match=r'No \[networks\] declared'):
        hg.validate(hg.load(path))
    with pytest.raises(ModelFileError, match=r'No \[networks\] declared'):
        hg.prepare(hg.load(path), tmp_path / 'run')
    if text.startswith('format'):
        with pytest.raises(ModelFileError, match=r'No \[networks\] declared'):
            load_config(path)
    if '[training]' in text:
        with pytest.raises(ModelFileError, match=r'\[training\] is an internal resolved-recipe table'):
            hg.validate(hg.load(path))


def test_format_less_data_only_file_still_loads_for_data_check(tmp_path):
    path = tmp_path / 'data.toml'
    path.write_text('[data]\nfactory = "gaussian_grid"\n', encoding='utf-8')
    assert load_config(path)['data']['factory'] == 'gaussian_grid'


def test_partial_base_files_still_load_for_inheritance(tmp_path):
    base = tmp_path / 'base.toml'
    base.write_text('format = "hypergan-model/1"\n[train]\nsteps = 9\n', encoding='utf-8')
    assert hg.load(base).spec['train']['steps'] == 9
    child = tmp_path / 'child.toml'
    child.write_text('extends = "gaussian-grid"\n[train]\nsteps = 7\n', encoding='utf-8')
    assert load_config(child)['training']['steps'] == 7


def test_lower_leaves_the_callers_spec_unchanged(tmp_path):
    spec = tomllib.loads(hg.save(hg.packaged('gaussian-grid'), tmp_path / 'a.toml').read_text(encoding='utf-8'))
    spec['dataset'] = spec.pop('data')
    model = hg.from_dict(spec, base=tmp_path)
    assert hg.lower(model)['data']['factory'] == 'gaussian_grid'
    assert 'data' not in model.spec
    saved = tomllib.loads(hg.save(model, tmp_path / 'b.toml').read_text(encoding='utf-8'))
    assert 'dataset' in saved and 'data' not in saved


def _workspace(tmp_path):
    (tmp_path / 'hypergan.toml').write_text(
        f'[workspace]\nruns_root = "runs"\n[paths]\ndata = "{tmp_path / "datasets"}"\n', encoding='utf-8')
    return load_workspace(tmp_path / 'hypergan.toml')


def test_workspace_interpolation_touches_only_path_fields(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    monkeypatch.setenv('HOME_DATA', '/from/the/environment')
    spec = {
        'description': 'loss on {data} set ${data}',
        'data': {'builtin': 'image_folder', 'args': {'root': '${paths.data}/cats', 'cache_dir': '${data}/cache'}},
        'networks': {'generator': {'source': 'linear(${width})', 'hndl': '${root}/nets/g.hndl',
                                   'args': {'networks': {'block': 'attention(h, ${heads})'},
                                            'network_files': {'head': '${workspace.root}/head.hndl'}}}},
        'evaluations': {'fid': {'label': '{data} FID'}},
    }
    result = interpolate_model_paths(spec, ws)
    data = str(tmp_path / 'datasets')
    assert result['data']['args'] == {'root': f'{data}/cats', 'cache_dir': f'{data}/cache'}
    generator = result['networks']['generator']
    assert generator['hndl'] == f'{tmp_path}/nets/g.hndl'
    assert generator['args']['network_files']['head'] == f'{tmp_path}/head.hndl'
    assert generator['source'] == 'linear(${width})'
    assert generator['args']['networks']['block'] == 'attention(h, ${heads})'
    assert result['description'] == spec['description']
    assert result['evaluations']['fid']['label'] == '{data} FID'
    with pytest.raises(ValueError, match=r"data\.args\.root: undefined workspace variable '\$\{HOME_DATA\}'"):
        interpolate_model_paths({'data': {'args': {'root': '${HOME_DATA}/x'}}}, ws)


def test_hndl_template_parameters_load_inside_a_workspace(tmp_path):
    _workspace(tmp_path)
    model = tmp_path / 'configs' / 'model.toml'
    model.parent.mkdir()
    model.write_text('format = "hypergan-model/1"\ndescription = "uses {data}"\n'
                     '[networks.generator.args.networks]\nblock = "attention(h, ${heads})"\n', encoding='utf-8')
    spec = hg.load(model).spec
    assert spec['networks']['generator']['args']['networks']['block'] == 'attention(h, ${heads})'
    assert spec['description'] == 'uses {data}'


def test_default_run_dir_is_a_fresh_slug_unless_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first, second = default_run_dir('model'), default_run_dir('model')
    assert first.parent == second.parent == tmp_path.resolve() and first != second
    assert re.fullmatch(r'model-[a-z]+-[a-z]+', first.name)
    assert default_run_dir('model', name='reference/100gaussians') == tmp_path.resolve() / 'reference-100gaussians'
    _workspace(tmp_path)
    (tmp_path / 'hypergan.toml').write_text(
        '[workspace]\nruns_root = "runs"\nrun_dir = "${runs_root}/train-${name}"\n', encoding='utf-8')
    named = default_run_dir('model', name='exp', start=tmp_path / 'm.toml')
    assert named == (tmp_path / 'runs' / 'train-exp').resolve()


def test_api_and_cli_share_the_run_directory_rule(tmp_path, monkeypatch):
    from hypergan.api import _default_run_dir
    monkeypatch.chdir(tmp_path)
    model = hg.packaged('gaussian-grid')  # named "reference/100gaussians", which is not a directory
    first, second = _default_run_dir(model), _default_run_dir(model)
    assert first.parent == tmp_path.resolve() and first != second
    assert first.name.startswith('gaussian-grid-')


def test_documented_network_fields_are_accepted():
    text = (ROOT / 'docs' / 'configuration.md').read_text(encoding='utf-8')
    table = text.split('### Network Specification Fields', 1)[1].split('\n\n', 2)[1]
    documented = set()
    for row in table.splitlines()[2:]:
        documented |= set(re.findall(r'`([a-z_]+)`', row.split('|')[1]))
    assert documented and documented <= NETWORK_FIELDS, documented - NETWORK_FIELDS
