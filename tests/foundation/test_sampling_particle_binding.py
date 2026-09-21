from copy import deepcopy

import pytest

from hypergan.config import DEFAULT, config_values, resolve_config


def test_sampling_binding_is_optional_without_changing_existing_defaults():
    assert config_values(resolve_config({})) == DEFAULT
    raw = deepcopy(DEFAULT)
    raw['components']['generator']['inputs']['x'] = 'components.encoder.latent'
    raw['components']['encoder'] = {
        'factory': 'example:Encoder', 'inputs': {'x': 'batch.gray'}}
    raw['sampling']['particle_ids'] = 'components.encoder.ids'
    assert resolve_config(raw)['sampling']['particle_ids'] == 'components.encoder.ids'


@pytest.mark.parametrize('binding', [None, 1, '', 'latent', 'components.missing.ids',
                                     'components.discriminator.ids', 'components.generator.'])
def test_sampling_binding_requires_generator_dependency_output(binding):
    with pytest.raises(ValueError, match='sampling.particle_ids'):
        resolve_config({'sampling': {'particle_ids': binding}})
