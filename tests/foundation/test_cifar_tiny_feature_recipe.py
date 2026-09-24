"""The simplified CIFAR pairing preserves the established training settings."""
from copy import deepcopy
from pathlib import Path
import tomllib


def test_recipe_changes_networks_and_removes_encoder_reconstruction():
    examples = Path(__file__).parents[2] / 'examples'
    baseline = tomllib.loads((examples / 'cifar-transgan.toml').read_text())
    actual = tomllib.loads((examples / 'cifar-tiny-transformer-resnet-features.toml').read_text())
    expected = deepcopy(baseline)
    expected['name'] = 'images/cifar-tiny-transformer-resnet-features'
    expected['components']['generator']['args']['file'] = 'networks/tiny-transformer-generator-32.hndl'
    expected['components']['discriminator']['args']['file'] = 'networks/resnet18-features-discriminator-32.hndl'
    del expected['components']['encoder'], expected['components']['reconstruction'], expected['objectives']
    assert actual == expected
