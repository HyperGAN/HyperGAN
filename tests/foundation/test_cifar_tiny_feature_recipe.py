"""The simplified CIFAR pairing preserves the established training settings."""
from copy import deepcopy
from pathlib import Path
import tomllib


def test_recipe_changes_networks_and_removes_encoder_reconstruction():
    examples = Path(__file__).parents[2] / 'examples'
    baseline_path = examples / 'cifar-transgan.toml'
    if not baseline_path.exists():
        baseline_path = examples / 'archive' / 'cifar-transgan.toml'
    actual_path = examples / 'cifar-tiny-transformer-resnet-features.toml'
    if not actual_path.exists():
        actual_path = examples / 'archive' / 'cifar-tiny-transformer-resnet-features.toml'
    baseline = tomllib.loads(baseline_path.read_text())
    actual = tomllib.loads(actual_path.read_text())
    expected = deepcopy(baseline)
    expected['name'] = 'images/cifar-tiny-transformer-resnet-features'
    expected['networks']['generator']['hndl'] = 'networks/tiny-transformer-generator-32.hndl'
    expected['networks']['discriminator']['hndl'] = 'networks/resnet18-features-discriminator-32.hndl'
    del expected['networks']['encoder'], expected['networks']['reconstruction']
    expected['losses'] = [loss for loss in expected['losses'] if loss.get('id') != 'reconstruction']
    assert actual == expected
