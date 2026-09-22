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
