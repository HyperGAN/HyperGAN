"""Keep the generator comparison on the same DINOv3 training recipe."""
from pathlib import Path

from hypergan.config import config_values, fingerprint, load_config


def test_transgan_changes_only_generator_and_run_name():
    examples = Path(__file__).parents[2] / 'examples'
    baseline = load_config(examples / 'sagan-adain-dinov3-multidepth-128.toml')
    variant = load_config(examples / 'transgan-dinov3-multidepth-128.toml')
    assert fingerprint(baseline) != fingerprint(variant)
    left, right = config_values(baseline), config_values(variant)
    assert left.pop('name') != right.pop('name')
    left_generator = left['components'].pop('generator')
    right_generator = right['components'].pop('generator')
    assert left == right
    assert left_generator['args'].pop('source') != right_generator['args'].pop('source')
    assert left_generator == right_generator


def test_cifar_transgan_preserves_discriminator_encoder_and_training_recipe():
    examples = Path(__file__).parents[2] / 'examples'
    baseline = load_config(examples / 'cifar-pretrained-sagan.toml')
    variant = load_config(examples / 'cifar-transgan.toml')
    assert fingerprint(baseline) != fingerprint(variant)
    left, right = config_values(baseline), config_values(variant)
    assert left.pop('name') != right.pop('name')
    baseline_generator = left['components'].pop('generator')
    generator = right['components'].pop('generator')
    assert left == right
    assert baseline_generator['factory'] == 'hypergan.image_components:CIFARGenerator'
    assert generator['factory'] == 'hndl'
    assert generator['inputs'] == {'z': 'latent'}
    assert generator['args']['input_shape'] == ['B', 64]
    assert generator['args']['output_shape'] == ['B', 3, 32, 32]
    assert generator['args']['source'] == (examples / 'networks/transgan-generator-32.hndl').read_text()
    assert right['prior']['args']['z_dim'] == right['components']['encoder']['args']['z_dim'] == 64
    assert right['components']['reconstruction']['reuse'] == 'generator'
    assert right['components']['reconstruction']['freeze_parameters']
    # Existing adapters snapshot their editable HNDL networks, including the
    # pretrained stages, rather than hiding a second architecture in the recipe.
    discriminator = right['components']['discriminator']['args']['networks']
    assert {'image_pixel', 'image_feature_head', 'image_resnet_stage1',
            'image_resnet_stage2', 'image_resnet_stage3', 'image_critic_score'} <= discriminator.keys()
    assert right['components']['encoder']['args']['networks']['image_encoder'].strip()
