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
    expected['networks']['generator']['hndl'] = 'networks/tiny-transformer-generator-32.hndl'
    expected['networks']['discriminator']['hndl'] = 'networks/resnet18-features-discriminator-32.hndl'
    del expected['networks']['encoder'], expected['networks']['reconstruction']
    expected['losses'] = [loss for loss in expected['losses'] if loss.get('id') != 'reconstruction']
    assert actual == expected
