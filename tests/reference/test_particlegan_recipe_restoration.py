"""Saved host recipes work before and after ParticleGAN's constructor guard."""
from types import SimpleNamespace

import particlegan

from hypergan.particlegan_integration import restore_particlegan_recipe


def test_compact_saved_recipe_keeps_its_recorded_formulation():
    saved = particlegan.get_recipe('bispf_optimized', batch_size=16).to_dict()
    restored = restore_particlegan_recipe(saved)
    assert restored.to_dict() == saved
    assert restored.critic_formulation == 'bispf'
    assert restored.ema_decay == 0

    saved_rhbi = particlegan.get_recipe('rhbi_optimized', batch_size=16).to_dict()
    restored_rhbi = restore_particlegan_recipe(saved_rhbi)
    assert restored_rhbi.to_dict() == saved_rhbi
    assert restored_rhbi.critic_formulation == 'rhbi'
    assert restored_rhbi.rhbi_weight == 1.0

    saved_v11 = particlegan.get_recipe('unified_v11', batch_size=16).to_dict()
    restored_v11 = restore_particlegan_recipe(saved_v11)
    assert restored_v11.to_dict() == saved_v11
    assert restored_v11.critic_formulation == 'unified_v11'
    assert restored_v11.optimizer_family == 'unified_v11_equil'
    assert restored_v11.prior_regularizer == 'unified_v11'

    saved_v12 = particlegan.get_recipe('unified_v12', batch_size=16).to_dict()
    restored_v12 = restore_particlegan_recipe(saved_v12)
    assert restored_v12.to_dict() == saved_v12
    assert restored_v12.critic_formulation == 'unified_v11'
    assert restored_v12.optimizer_family == 'unified_v12_equil'
    assert restored_v12.prior_regularizer == 'unified_v11'


def test_modern_restoration_never_calls_a_partial_constructor(monkeypatch):
    saved = {'lr': 0.003}
    calls = []
    sentinel = object()

    def restore(fields):
        calls.append(fields)
        return sentinel

    monkeypatch.setattr(particlegan, 'Recipe', SimpleNamespace(from_dict=restore))
    assert restore_particlegan_recipe(saved) is sentinel
    assert calls == [saved]


def test_094_constructor_fallback_keeps_all_recorded_fields(monkeypatch):
    class LegacyRecipe:
        def __init__(self, **fields):
            self.fields = fields

    monkeypatch.setattr(particlegan, 'Recipe', LegacyRecipe)
    saved = {'name': 'ka2', 'lr': 0.003, 'ema_decay': 0.995}
    assert restore_particlegan_recipe(saved).fields == saved
