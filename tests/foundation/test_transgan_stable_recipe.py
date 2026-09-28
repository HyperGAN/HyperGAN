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


def test_e2_recipe_changes_only_generator_and_run_name():
    examples = Path(__file__).parents[2] / 'examples'
    baseline = tomllib.loads((examples / 'transgan-resnet-multiscale-128-stable.toml').read_text())
    e2 = tomllib.loads((examples / 'transgan-resnet-multiscale-128-e2.toml').read_text())
    expected = deepcopy(baseline)
    expected['name'] = 'images/logos-transgan-resnet-multiscale-128-e2'
    expected['components']['generator']['args']['file'] = 'networks/transgan-generator-128-e2.hndl'
    assert e2 == expected


def test_e3_recipe_changes_only_generator_and_run_name():
    examples = Path(__file__).parents[2] / 'examples'
    baseline = tomllib.loads((examples / 'transgan-resnet-multiscale-128-stable.toml').read_text())
    e3 = tomllib.loads((examples / 'transgan-resnet-multiscale-128-e3.toml').read_text())
    expected = deepcopy(baseline)
    expected['name'] = 'images/logos-transgan-resnet-multiscale-128-e3'
    expected['components']['generator']['args']['file'] = 'networks/transgan-generator-128-e3.hndl'
    assert e3 == expected


def test_equalized_recipe_changes_only_generator_and_run_name():
    examples = Path(__file__).parents[2] / 'examples'
    baseline = tomllib.loads((examples / 'transgan-resnet-multiscale-128-stable.toml').read_text())
    equalized = tomllib.loads((examples / 'transgan-resnet-multiscale-128-equalized.toml').read_text())
    expected = deepcopy(baseline)
    expected['name'] = 'images/logos-transgan-resnet-multiscale-128-equalized'
    expected['components']['generator']['args']['file'] = 'networks/transgan-generator-128-equalized.hndl'
    assert equalized == expected


def test_simple_styletransformer_preserves_d_and_prior_rates():
    import pytest

    examples = Path(__file__).parents[2] / 'examples'
    baseline = tomllib.loads((examples / 'transgan-resnet-multiscale-128-stable.toml').read_text())
    simple = tomllib.loads((examples / 'simple-styletransformer-resnet-128-low-g-lr.toml').read_text())
    expected = deepcopy(baseline)
    expected['name'] = 'images/logos-simple-styletransformer-resnet-128-low-g-lr'
    expected['components']['generator']['args']['file'] = 'networks/simple-styletransformer-generator-128.hndl'
    expected['optimizer'].update(lr=.00002, d_lr_mult=10.0, prior_lr_mult=100.0)
    assert simple == expected
    opt = simple['optimizer']
    assert opt['lr'] == pytest.approx(baseline['optimizer']['lr'] / 10)
    for multiplier in ('d_lr_mult', 'prior_lr_mult'):
        assert opt['lr'] * opt[multiplier] == pytest.approx(
            baseline['optimizer']['lr'] * baseline['optimizer'][multiplier])
