"""The stable network experiment preserves the existing training recipe."""
from copy import deepcopy
from pathlib import Path
import tomllib


def test_stable_recipe_changes_only_explicit_network_settings():
    examples = Path(__file__).parents[2] / 'examples'
    baseline = tomllib.loads((examples / 'transgan-dinov3-multidepth-128.toml').read_text())
    stable = tomllib.loads((examples / 'transgan-projected-dinov3-128-stable.toml').read_text())
    expected = deepcopy(baseline)
    expected['name'] = 'images/logos-transgan-projected-dinov3-128-stable'
    expected['components']['generator']['args'].update(
        input_shape=['B', 512], file='networks/transgan-generator-128-stable.hndl')
    expected['components']['discriminator']['args'].update(
        output_shape=['B', 4, 4, 4], file='networks/dinov3-projected-discriminator-128-stable.hndl')
    expected['prior']['args']['z_dim'] = 512
    # Make the existing default explicit: stochastic augmentation must not use
    # finite differences across independently sampled transforms.
    expected['gradient_penalty']['method'] = 'autograd'
    assert stable == expected
