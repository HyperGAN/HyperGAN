"""Execution-profile checks that need the train extra (ParticleGAN presets).

Moved out of tests/foundation, which runs without Torch or ParticleGAN.
"""
from hypergan.config import resolve_config


def test_replicated_recipe_allows_particlegan_presets_and_recipe_optimizer():
    """Verify validate_replicated_recipe accepts resolved presets and optimizer.implementation='recipe'."""
    from hypergan.execution_profiles import validate_replicated_recipe
    preset_config = resolve_config({
        "defaults": "particlegan",
        "training": {"device": "cpu", "batch_size": 16},
    })
    assert preset_config.get("recipe")
    assert preset_config["optimizer"]["implementation"] == "recipe"
    validate_replicated_recipe(preset_config)
