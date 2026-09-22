"""Persist measured startup learning rates before the first durable checkpoint."""
import copy
import hashlib
from pathlib import Path

from .config import config_values, fingerprint
from .run_state import atomic_json, sync_directory

METHOD = 'measured-update-response'


def tune_initialized(trainer, run_dir, on_event=None):
    """Publish rate overrides after disposable trials have restored step zero.

    The source config and initialized weights are unchanged. The first full
    checkpoint stores the selected rates; resume never repeats calibration.
    The historical metadata key ``initialization_tuning`` remains readable.
    """
    from .startup_dynamics import tune_startup_dynamics
    from .tuning_overrides import optimizer_lr_override
    from .provenance import hypergan_source

    if trainer.step != 0 or trainer.opt_g.state or trainer.opt_d.state:
        raise ValueError('Startup tuning requires untouched initialization and empty optimizer state')
    if getattr(trainer, 'g_lr_warmup', None) is not None:
        raise ValueError('Startup tuning requires initialization without an existing warmup schedule')
    root = Path(run_dir) / 'tuning'
    root.mkdir(exist_ok=False)
    sync_directory(root.parent)
    atomic_json(root / 'config.base.json', config_values(trainer.config))
    original_base_lrs = copy.deepcopy(trainer.base_lrs)
    original_lrs = [[group['lr'] for group in optimizer.param_groups]
                    for optimizer in (trainer.opt_g, trainer.opt_d)]

    def progress(value):
        if on_event is not None:
            on_event(dict(value, phase='dynamics', method=METHOD))

    try:
        dynamics = tune_startup_dynamics(trainer, progress=progress)
        if (trainer.step != 0 or trainer.opt_g.state or trainer.opt_d.state
                or trainer.base_lrs != original_base_lrs
                or [[group['lr'] for group in optimizer.param_groups]
                    for optimizer in (trainer.opt_g, trainer.opt_d)] != original_lrs):
            raise ValueError('Startup dynamics trials did not restore the original training boundary')
        override = optimizer_lr_override(trainer.config, dynamics['selected_g_lr_factor'],
                                         dynamics.get('selected_d_lr_factor', 1.0))
        outcome = dynamics['outcome']
        if (outcome not in ('selected', 'kept_baseline', 'unresolved', 'skipped')
                or (override['factor'] != 1 or override['d_factor'] != 1) != (outcome == 'selected')):
            raise ValueError('Startup dynamics factor differs from its recorded selection outcome')
        trainer.opt_g.param_groups[0]['lr'] = override['effective_g_lr']
        trainer.base_lrs[0][0] = override['effective_g_lr']
        trainer.opt_d.param_groups[0]['lr'] = override['effective_d_lr']
        trainer.base_lrs[1][0] = override['effective_d_lr']
        report = {
            'schema_version': 2, 'kind': 'hypergan-startup-rate-tuning', 'method': METHOD,
            'outcome': outcome, 'retained_training_updates': 0,
            'disposable_trial_updates': dynamics.get('disposable_completed_updates', 0),
            'dynamics': dynamics, 'optimizer_override': override, 'source': hypergan_source(),
            'schedule': 'Selected base rates follow the configured annealing schedule; no startup ramp back to source rates.',
        }
        overrides = {
            'schema_version': 2, 'kind': 'hypergan-startup-rate-overrides', 'method': METHOD,
            'base_config_sha256': fingerprint(trainer.config), 'optimizer_override': override,
            'recovery': 'Selected rates are stored in the initial full training checkpoint; never replay on resume.',
        }
        atomic_json(root / 'overrides.json', overrides)
        atomic_json(root / 'report.json', report)
        return {
            'method': METHOD, 'outcome': outcome, 'retained_training_updates': 0,
            'disposable_trial_updates': report['disposable_trial_updates'],
            'dynamics_outcome': outcome, 'dynamics_reason': dynamics.get('reason'),
            'selected_g_lr_factor': override['factor'], 'selected_d_lr_factor': override['d_factor'],
            'optimizer_override': override,
            'message': ({
                'selected': f"Measured updates selected learning rates G x{override['factor']:g}, D x{override['d_factor']:g}",
                'kept_baseline': 'Measured startup checks retained configured G and D learning rates',
                'unresolved': 'Startup calibration unresolved; configured G and D learning rates retained',
                'skipped': 'Startup rate calibration skipped: ' + str(dynamics.get('reason', 'unsupported configuration')),
            }[outcome]),
            'base_config_path': str(root / 'config.base.json'),
            'base_config_sha256': fingerprint(trainer.config),
            'overrides_path': str(root / 'overrides.json'),
            'overrides_sha256': hashlib.sha256((root / 'overrides.json').read_bytes()).hexdigest(),
            'report_path': str(root / 'report.json'),
            'report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest(),
        }
    except BaseException:
        # Trial rollback is owned by the tuner. Persistence may fail after rates
        # are installed; never leave those partially published rates applied.
        trainer.base_lrs = original_base_lrs
        for optimizer, rates in zip((trainer.opt_g, trainer.opt_d), original_lrs):
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate
        raise
