"""ParticleGAN adapter: resolve public defaults, restore recipes and bind host settings.

Formulation choices belong to ParticleGAN. Recipe snapshots are opaque dependency
state; HyperGAN maps its existing configuration fields and checks host features.
This module is safe to import without the numerical runtime.
"""
import functools
import importlib.util
from pathlib import Path
import sys

# `defaults = "particlegan"` fills these omitted fields from the installed
# ParticleGAN's current public preset instead of DEFAULT. Resolved configurations
# record the concrete values, so a later ParticleGAN release cannot silently
# change an existing run; its resume then fails the configuration check.
PARTICLEGAN_DEFAULT_FIELDS = {
    ("gradient_penalty", "coeff"): "reg_coeff",
    ("gradient_penalty", "kappa"): "reg_kappa",
    ("gradient_penalty", "lazy_k"): "reg_every",
    ("gradient_penalty", "anchor_weight"): "reg_anchor_weight",
    ("prior_regularizer", "weight"): "prior_reg",
    ("prior_regularizer", "target_std"): "prior_reg_target_std",
    ("prior_regularizer", "eps"): "prior_reg_eps",
    ("optimizer", "lr"): "lr",
    ("optimizer", "d_lr_mult"): "d_lr_mult",
    ("optimizer", "prior_lr_mult"): "prior_lr_mult",
    ("optimizer", "betas"): "betas",
    ("optimizer", "prior_betas"): "prior_betas",
    ("optimizer", "d_guard_ratio"): "d_guard_ratio",
    ("optimizer", "d_guard_min_steps"): "d_guard_min_steps",
    ("optimizer", "latent_damping_max_rate"): "latent_damping_max_rate",
    ("training", "ema"): "ema_decay",
    ("training", "lr_anneal_start"): "lr_anneal_start",
    ("training", "lr_floor"): "lr_floor",
    ("training", "network_lr_floor"): "network_lr_floor",
    ("training", "network_lr_horizon_cap"): "network_lr_horizon_cap",
    ("training", "input_noise_std"): "input_noise_std",
    ("training", "input_noise_anneal_end"): "input_noise_anneal_end",
    ("training", "output_noise_std"): "output_noise_std",
    ("training", "output_noise_warmup"): "output_noise_warmup",
}


@functools.cache
def particlegan_recipe_fields():
    """Read the installed public preset without importing Torch or ParticleGAN.

    Loads the public recipes module without importing the package, whose
    __init__ imports Torch; configuration loading stays Torch-free.
    """
    package = importlib.util.find_spec("particlegan")
    if package is None or not package.submodule_search_locations:
        raise ValueError('defaults = "particlegan" requires ParticleGAN (the train extra)')
    name = "_hypergan_particlegan_recipes"
    recipe_path = Path(package.submodule_search_locations[0]) / "recipes.py"
    spec = importlib.util.spec_from_file_location(name, recipe_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        recipe = module.get_recipe_fields()
    finally:
        del sys.modules[name]
    return recipe


def particlegan_defaults():
    recipe = dict(particlegan_recipe_fields())
    if recipe["prior_betas"] is None:
        recipe["prior_betas"] = recipe["betas"]
    return {key: list(recipe[field]) if isinstance(recipe[field], tuple) else recipe[field]
            for key, field in PARTICLEGAN_DEFAULT_FIELDS.items() if field in recipe}


def particlegan_recipe(config):
    """The ParticleGAN recipe holding this configuration's training formulation.

    The dependency owns the snapshot and all formulation choices. Existing scalar
    sections supply explicit overrides; unsnapshotted configurations use the
    dependency's public restoration defaults.
    """
    from particlegan import Recipe

    opt, penalty, training = config['optimizer'], config['gradient_penalty'], config['training']
    fields = dict(config.get('recipe') or {})
    fields.update(
        total_steps=training['steps'], batch_size=training['batch_size'],
        lr=opt['lr'], d_lr_mult=opt['d_lr_mult'], prior_lr_mult=opt['prior_lr_mult'],
        betas=tuple(opt['betas']), prior_betas=tuple(opt['prior_betas']),
        d_guard_ratio=opt['d_guard_ratio'], d_guard_min_steps=opt['d_guard_min_steps'],
        latent_damping_max_rate=opt['latent_damping_max_rate'],
        reg_coeff=penalty['coeff'], reg_kappa=penalty['kappa'], reg_every=penalty['lazy_k'],
        reg_anchor_weight=penalty['anchor_weight'],
        prior_reg=config['prior_regularizer']['weight'],
        prior_reg_target_std=config['prior_regularizer']['target_std'],
        prior_reg_eps=config['prior_regularizer']['eps'], ema_decay=training['ema'],
        lr_anneal_start=training['lr_anneal_start'], lr_floor=training['lr_floor'],
        network_lr_floor=training['network_lr_floor'], network_lr_horizon_cap=training['network_lr_horizon_cap'],
        input_noise_std=training['input_noise_std'], input_noise_anneal_end=training['input_noise_anneal_end'],
        output_noise_std=training['output_noise_std'], output_noise_warmup=training['output_noise_warmup'])
    recipe = Recipe(**fields)
    unsupported = {
        'continuous_policy': recipe.continuous_policy is not None,
        'particle_birth_death': recipe.particle_birth_death,
        'row_evidence_gate': recipe.row_evidence_gate,
        'serve_average': recipe.serve_average != 0,
        'output_noise_mode': recipe.output_noise_mode != 'fixed',
        'constraint_geometry_mode': recipe.constraint_geometry_mode not in ('none', 'nonascent', 'direction_blend'),
        'critic_step_mode': recipe.critic_step_mode != 'none',
        'prior_proximal_strength': recipe.prior_proximal_strength != 0,
        'kinetic_transport_weight': recipe.kinetic_transport_weight != 0,
        'kinetic_transport_local_weight': recipe.kinetic_transport_local_weight != 0,
        'beta2_end': recipe.beta2_end is not None,
        'reg_coeff_end': recipe.reg_coeff_end is not None,
        'model': recipe.model != 'gan',
        'conditioning': recipe.conditioning == 'ucd',
        'encoder_mode': recipe.encoder_mode != 'none',
        'row_policy': recipe.row_policy != 'independent',
    }
    rejected = [key for key, active in unsupported.items() if active]
    if rejected:
        raise ValueError('HyperGAN reference loop does not implement recipe mechanisms: ' + ', '.join(rejected))
    return recipe


def implementation_files():
    """Hash the dependency as a whole, including helpers its factories may use."""
    import hashlib
    import particlegan

    root = Path(particlegan.__file__).resolve().parent
    return {'particlegan.' + path.relative_to(root).with_suffix('').as_posix().replace('/', '.'):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))}
