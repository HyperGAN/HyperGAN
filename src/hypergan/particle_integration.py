"""Explicit develop integration; recorded leaders are experimental, not qualified."""
from dataclasses import asdict

PARTICLEGAN_COMMIT = "6189649bc3c3478d452fadafda7ae84e9c668564"
BCAP_PRESET = dict(critic_formulation='bcap', optimizer_family='dualnorm', reg_arm='b_cap',
                   reg_coeff=1., reg_kappa=1., reg_every=1, lr=.012, d_lr_mult=1.5, prior_lr_mult=2.5,
                   betas=(0.,.999), lr_floor=1., network_lr_floor=1., network_lr_horizon_cap=None,
                   beta2_end=None, reg_coeff_end=None, d_guard_ratio=0., reg_anchor_weight=0.,
                   latent_damping_max_rate=0., direct_particle_gain=False, prior_reg=0., ema_decay=0.,
                   input_noise_std=0., output_noise_std=0., output_noise_warmup=0., optimizer_momentum=0.,
                   loss='non_saturating', optimizer_smoothing=.001, optimizer_convolution='per_offset',
                   constraint_geometry_mode='direction_blend')
BISPF_LEADER = {'adam_variant': 'pytorch',
 'alpha_bar': [1.0, 0.9, 0.5, 0.05, 0.0001],
 'amsgrad': False,
 'beta2_anneal_end': 0.2,
 'beta2_end': None,
 'betas': [0.0, 0.999],
 'birth_death_backend': 'knn',
 'birth_death_cells': 64,
 'birth_death_chunk': 256,
 'birth_death_feature_scale': 'none',
 'birth_death_isolation': False,
 'birth_death_metric_rank': 8,
 'birth_death_parent_policy': 'real_anchor',
 'birth_death_space': 'data',
 'conditioning': 'scalar',
 'constraint_geometry_mode': 'direction_blend',
 'continuous_policy': None,
 'critic_formulation': 'bispf',
 'critic_payoff_damping': True,
 'critic_r1_real': True,
 'critic_step_mode': 'none',
 'd_betas': None,
 'd_eps': None,
 'd_guard_min_steps': 200,
 'd_guard_ratio': 0.0,
 'd_lr_mult': 1.5,
 'direct_particle_betas': [0.0, 0.9],
 'direct_particle_gain': False,
 'distance_reduction': 'sum',
 'ema_decay': 0.0,
 'encoder_mode': 'none',
 'eps': 1e-08,
 'input_noise_anneal_end': 0.1,
 'input_noise_std': 0.0,
 'kinetic_transport_local_weight': 0.0,
 'kinetic_transport_projections': 32,
 'kinetic_transport_weight': 0.0,
 'latent_damping_max_rate': 0.0,
 'loss': 'non_saturating',
 'loss_labels': [0.0, 1.0, 1.0],
 'lr': 0.012,
 'lr_anneal_start': 0.6,
 'lr_control': 'mobility',
 'lr_decay_rate': 0.96,
 'lr_decay_staircase': False,
 'lr_decay_steps': 50000,
 'lr_floor': 1.0,
 'lr_schedule': 'cosine',
 'model': 'gan',
 'name': 'bispf',
 'network_lr_floor': 1.0,
 'network_lr_horizon_cap': None,
 'num_classes': None,
 'observation_sigma': 0.03,
 'optimizer_adam_lr': None,
 'optimizer_convolution': 'per_offset',
 'optimizer_family': 'dualnorm',
 'optimizer_momentum': 0.0,
 'optimizer_smoothing': 0.001,
 'optimizer_svd_backend': 'native',
 'output_noise_mode': 'fixed',
 'output_noise_std': 0.0,
 'output_noise_warmup': 0.0,
 'particle_birth_death': False,
 'prior_betas': None,
 'prior_eps': None,
 'prior_l2': 0.0,
 'prior_lr_mult': 2.5,
 'prior_reg': 0.01,
 'prior_reg_eps': 0.0001,
 'prior_reg_target_std': 1.0,
 'prior_regularizer': 'sbnp',
 'prior_update': 'learned',
 'reconstruction_weight': 1.0,
 'reg_anchor_min_decay': 0.9,
 'reg_anchor_weight': 0.0,
 'reg_arm': None,
 'reg_coeff': 1.0,
 'reg_coeff_anneal_end': 0.2,
 'reg_coeff_end': None,
 'reg_every': 1,
 'reg_kappa': 1.0,
 'reopen_anchor': 'hold',
 'reopen_guard': None,
 'reopen_signal': 'data',
 'routing_temperature': 0.25,
 'row_evidence_exclude': True,
 'row_evidence_gate': False,
 'row_evidence_hold': True,
 'row_evidence_hot': True,
 'row_evidence_null': 'theory',
 'row_policy': 'independent',
 'serve_average': 0.0,
 'table_release_rule': 'any',
 'ucd_target': 'class',
 'ucd_weight': 0.02}


def verify_source():
    import importlib.util
    import importlib.metadata
    import json
    from pathlib import Path
    import subprocess
    import os
    fds = tuple(int(x) for x in os.environ.get("ALZ_LEASE_FDS", "").split(",") if x)
    spec = importlib.util.find_spec('particlegan')
    path = Path(spec.origin).resolve() if spec and spec.origin else None
    commit = None
    if path is not None:
        try:
            commit = subprocess.check_output(['git','-C',str(path.parent),'rev-parse','HEAD'],
                                             text=True, stderr=subprocess.DEVNULL, timeout=5, pass_fds=fds).strip()
            dirty = subprocess.check_output(['git','-C',str(path.parent),'status','--porcelain',
                                             '--untracked-files=no','--','.'], text=True, timeout=5, pass_fds=fds).strip()
            if dirty: raise ValueError('ParticleGAN package has tracked modifications')
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                direct = json.loads(importlib.metadata.distribution('particlegan').read_text('direct_url.json') or '{}')
                commit = direct.get('vcs_info',{}).get('commit_id')
            except importlib.metadata.PackageNotFoundError:
                pass
    if commit != PARTICLEGAN_COMMIT:
        raise ValueError(f'ParticleGAN source is not pinned: expected {PARTICLEGAN_COMMIT}, got {commit}')
    return {'commit':commit, 'module':str(path)}


def make_recipe(trainer, **host):
    verify_source()
    from particlegan import get_recipe
    if trainer == "bispf":
        from particlegan import Recipe
        return Recipe(**{**BISPF_LEADER, **host})
    if trainer not in ("ka2", "k3p", "bcap"):
        raise ValueError("unsupported ParticleGAN trainer")
    return get_recipe("gan", trainer=trainer, **host)


def receipt(recipe):
    import json
    return {"repository": "https://github.com/255BITS/ParticleGAN", "commit": PARTICLEGAN_COMMIT,
            "loaded_source": verify_source(), "recipe": json.loads(json.dumps(asdict(recipe))), "qualification": "unqualified host adaptation"}

# Pure-data snapshot of Recipe defaults at PARTICLEGAN_COMMIT.
# Configuration and model inspection must work without the train extra.
PINNED_RECIPE_DEFAULTS = {'adam_variant': 'pytorch',
 'alpha_bar': (1.0, 0.9, 0.5, 0.05, 0.0001),
 'amsgrad': False,
 'batch_size': 2048,
 'beta2_anneal_end': 0.2,
 'beta2_end': None,
 'betas': (0.0, 0.999),
 'birth_death_backend': 'knn',
 'birth_death_cells': 64,
 'birth_death_chunk': 256,
 'birth_death_feature_scale': 'none',
 'birth_death_isolation': False,
 'birth_death_metric_rank': 8,
 'birth_death_parent_policy': 'real_anchor',
 'birth_death_space': 'data',
 'conditioning': 'scalar',
 'constraint_geometry_mode': 'none',
 'continuous_policy': None,
 'critic_formulation': 'ka2',
 'critic_payoff_damping': True,
 'critic_r1_real': True,
 'critic_step_mode': 'none',
 'd_betas': None,
 'd_eps': None,
 'd_guard_min_steps': 200,
 'd_guard_ratio': 5.0,
 'd_lr_mult': 1.0,
 'direct_particle_betas': (0.0, 0.9),
 'direct_particle_gain': True,
 'distance_reduction': 'sum',
 'ema_decay': 0.995,
 'encoder_mode': 'none',
 'eps': 1e-08,
 'input_noise_anneal_end': 0.1,
 'input_noise_std': 0.5,
 'kinetic_transport_local_weight': 0.0,
 'kinetic_transport_projections': 32,
 'kinetic_transport_weight': 0.0,
 'latent_damping_max_rate': 0.5,
 'loss': 'relativistic',
 'loss_labels': (0.0, 1.0, 1.0),
 'lr': 0.00425,
 'lr_anneal_start': 0.6,
 'lr_control': 'mobility',
 'lr_decay_rate': 0.96,
 'lr_decay_staircase': False,
 'lr_decay_steps': 50000,
 'lr_floor': 0.05,
 'lr_schedule': 'cosine',
 'model': 'gan',
 'name': 'ka2',
 'network_lr_floor': 0.01,
 'network_lr_horizon_cap': 1600,
 'num_classes': None,
 'num_particles': 20000,
 'observation_sigma': 0.03,
 'optimizer_adam_lr': None,
 'optimizer_convolution': 'none',
 'optimizer_family': 'formulation',
 'optimizer_momentum': 0.0,
 'optimizer_smoothing': 0.0,
 'optimizer_svd_backend': 'native',
 'output_noise_mode': 'fixed',
 'output_noise_std': 0.029,
 'output_noise_warmup': 0.2,
 'particle_birth_death': False,
 'prior_betas': None,
 'prior_coulomb_confinement': 0.35,
 'prior_coulomb_neighbors': 8,
 'prior_coulomb_softening': 0.1,
 'prior_eps': None,
 'prior_kind': 'particles',
 'prior_l2': 0.0,
 'prior_lr_mult': 2.0,
 'prior_proximal_strength': 0.0,
 'prior_reg': 0.0,
 'prior_reg_eps': 0.0001,
 'prior_reg_target_std': 1.0,
 'prior_regularizer': 'vicreg',
 'prior_sigreg_knots': 17,
 'prior_sigreg_projections': 32,
 'prior_update': 'learned',
 'reconstruction_weight': 1.0,
 'reg_anchor_min_decay': 0.9,
 'reg_anchor_weight': 1.0,
 'reg_arm': None,
 'reg_coeff': 1.0,
 'reg_coeff_anneal_end': 0.2,
 'reg_coeff_end': None,
 'reg_every': 1,
 'reg_kappa': 1.0,
 'reopen_anchor': 'hold',
 'reopen_guard': None,
 'reopen_signal': 'data',
 'routing_temperature': 0.25,
 'row_evidence_exclude': True,
 'row_evidence_gate': False,
 'row_evidence_hold': True,
 'row_evidence_hot': True,
 'row_evidence_null': 'theory',
 'row_policy': 'independent',
 'serve_average': 0.0,
 'sigma_rel': 0.0,
 'standardize': True,
 'table_release_rule': 'any',
 'total_steps': 7000,
 'ucd_target': 'class',
 'ucd_weight': 0.02,
 'z_dim': 2}
