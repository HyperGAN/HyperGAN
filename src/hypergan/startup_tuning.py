"""Persist an owned initialization search before the first durable checkpoint."""
import hashlib
import copy
from pathlib import Path

from .config import config_values, fingerprint
from .run_state import atomic_json, sync_directory


def tune_initialized(trainer, run_dir, on_event=None):
    """Publish run-local provenance and synchronize only changed owned EMA weights.

    The normal step-zero training checkpoint stores the selected tensors and all
    RNG/data state. These JSON artifacts explain that checkpoint; they are never
    replayed on resume and the user's source configuration stays the baseline.
    """
    import torch
    from .initialization_tuning import tune_initialization
    from .startup_dynamics import tune_startup_dynamics
    from .tuning_overrides import generator_lr_override
    from .provenance import hypergan_source

    if trainer.step != 0 or trainer.opt_g.state or trainer.opt_d.state:
        raise ValueError('Startup tuning requires untouched initialization and empty optimizer state')
    root = Path(run_dir) / 'tuning'
    root.mkdir(exist_ok=False)
    sync_directory(root.parent)
    baseline = config_values(trainer.config)
    atomic_json(root / 'config.base.json', baseline)
    owned = {id(parameter) for parameter in trainer.program.generator_parameters}
    parameters = {name: parameter for name, parameter in trainer.graph.named_parameters()
                  if id(parameter) in owned}
    ema = dict(trainer.ema_graph.named_parameters())
    originals = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    ema_originals = {name: ema[name].detach().cpu().clone() for name in parameters}
    original_base_lrs = copy.deepcopy(trainer.base_lrs)
    original_lrs = [[group['lr'] for group in optimizer.param_groups] for optimizer in (trainer.opt_g, trainer.opt_d)]
    def progress(phase):
        return (lambda value: on_event(dict(value, phase=phase))) if on_event is not None else None
    try:
        report = tune_initialization(trainer, progress=progress('initialization'))
        changed = []
        with torch.no_grad():
            for name, parameter in parameters.items():
                if not torch.equal(parameter.detach().cpu(), originals[name]):
                    ema[name].copy_(parameter)
                    changed.append(name)
        report['ema_synchronized_parameters'] = changed
        dynamics = tune_startup_dynamics(trainer, progress=progress('dynamics'))
        if (trainer.step != 0 or trainer.opt_g.state or trainer.opt_d.state
                or trainer.base_lrs != original_base_lrs
                or [[group['lr'] for group in optimizer.param_groups] for optimizer in (trainer.opt_g, trainer.opt_d)] != original_lrs):
            raise ValueError('Startup dynamics trials did not restore the original training boundary')
        override = generator_lr_override(trainer.config, dynamics['selected_g_lr_factor'])
        outcome = dynamics['outcome']
        if (outcome not in ('selected', 'kept_baseline', 'unresolved', 'skipped')
                or (override['factor'] != 1) != (outcome == 'selected')):
            raise ValueError('Startup dynamics factor differs from its recorded selection outcome')
        trainer.opt_g.param_groups[0]['lr'] = override['effective_g_lr']
        trainer.base_lrs[0][0] = override['effective_g_lr']
        report['initialization_optimizer_steps'] = report.pop('optimizer_steps', 0)
        report['retained_training_updates'] = 0
        report['disposable_trial_updates'] = dynamics.get('disposable_completed_updates', 0)
        report['dynamics'] = dynamics
        report['optimizer_override'] = override
        selected_digest = hashlib.sha256()
        for name, parameter in sorted(parameters.items()):
            value = parameter.detach().cpu().contiguous()
            selected_digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}:'.encode())
            selected_digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        report['selected_parameters_sha256'] = selected_digest.hexdigest()
        report['source'] = hypergan_source()
        overrides = {
            'schema_version': 1,
            'kind': 'hypergan-startup-initialization-overrides',
            'base_config_sha256': fingerprint(trainer.config),
            'selected_candidate': report['selected_candidate'],
            'transformations': report['transformations'],
            'optimizer_override': override,
            'recovery': 'Selected tensors are stored in the initial full training checkpoint; never replay on resume.',
        }
        atomic_json(root / 'overrides.json', overrides)
        atomic_json(root / 'report.json', report)
        return {
            'selected_parameters_sha256': report['selected_parameters_sha256'],
            'outcome': report['outcome'],
            'selected_candidate': report['selected_candidate'],
            'retained_training_updates': 0,
            'disposable_trial_updates': report['disposable_trial_updates'],
            'dynamics_outcome': outcome,
            'dynamics_reason': dynamics.get('reason'),
            'selected_g_lr_factor': override['factor'],
            'optimizer_override': override,
            'message': ({'selected': f"Startup checks selected generator learning rate x{override['factor']:g}",
                         'kept_baseline': 'Startup checks passed; configured generator learning rate retained',
                         'unresolved': 'Startup checks unresolved; configured generator learning rate retained',
                         'skipped': 'Startup initialization checked; dynamics calibration skipped'}[outcome]),
            'base_config_path': str(root / 'config.base.json'),
            'base_config_sha256': fingerprint(trainer.config),
            'overrides_path': str(root / 'overrides.json'),
            'overrides_sha256': hashlib.sha256((root / 'overrides.json').read_bytes()).hexdigest(),
            'report_path': str(root / 'report.json'),
            'report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest(),
        }
    except BaseException:
        # Includes artifact I/O failure after a candidate was selected. Never
        # leave a partially applied initialization available to the trainer.
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(originals[name])
                ema[name].copy_(ema_originals[name])
        trainer.base_lrs = original_base_lrs
        for optimizer, rates in zip((trainer.opt_g, trainer.opt_d), original_lrs):
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate
        raise
