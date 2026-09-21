"""Architecture text is part of the immutable numerical recipe."""
from copy import deepcopy

import pytest

from hypergan.config import DEFAULT, config_values, fingerprint, load_config, resolve_config
from hypergan.network_config import render_source


def test_relative_network_file_is_snapshotted_and_edits_change_identity(tmp_path):
    recipe = tmp_path / 'recipe.toml'
    architecture = tmp_path / 'generator.hndl'
    architecture.write_text('linear(8)\nrelu()\nlinear()\n')
    recipe.write_text('''[components.generator]
factory = "hndl"
inputs = {x = "latent"}
[components.generator.args]
file = "generator.hndl"
input_shape = ["B", 4]
output_shape = ["B", 2]
[components.discriminator]
factory = "hndl"
inputs = {x = "candidate"}
[components.discriminator.args]
source = "linear()"
input_shape = ["B", 2]
output_shape = ["B", 1]
''')
    before = load_config(recipe)
    architecture.write_text('linear(16)\nrelu()\nlinear()\n')
    assert fingerprint(load_config(recipe)) != fingerprint(before)
    architecture.unlink()
    assert fingerprint(resolve_config(config_values(before))) == fingerprint(before)


def test_packaged_image_templates_are_recorded_without_torch():
    raw = deepcopy(DEFAULT)
    raw['components']['generator'] = {'factory': 'hypergan.image_components:CIFARGenerator',
                                     'args': {}, 'inputs': {'z': 'latent'}}
    config = resolve_config(raw)
    sources = config['components']['generator']['args']['networks']
    assert {'image_generator', 'image_attention'} == set(sources)
    changed = deepcopy(config_values(config))
    changed['components']['generator']['args']['networks']['image_generator'] += '\n# change\n'
    assert fingerprint(resolve_config(changed)) != fingerprint(config)


@pytest.mark.parametrize('changes', [
    {'source': ''}, {'source': 4}, {'input_shape': [4]},
    {'output_shape': ['B', 0]}, {'output_shape': ['B', True]},
    {'input_shape': ['batch', 4]}, {'concat_inputs': ['x', 'x']},
    {'parameters': {'width': float('nan')}, 'source': 'linear(${width})'},
    {'concat_dim': 0}, {'unexpected': True},
])
def test_invalid_hndl_contracts_rejected(changes):
    raw = deepcopy(DEFAULT)
    raw['components']['generator']['args'].update(changes)
    with pytest.raises(ValueError):
        resolve_config(raw)


def test_string_parameters_cannot_inject_operations():
    assert render_source('pretrained(${path})', {'path': "x')\nlinear(9)\n#"}) == 'pretrained("x\')\\nlinear(9)\\n#")'
