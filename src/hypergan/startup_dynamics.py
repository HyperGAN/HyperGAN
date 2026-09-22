"""Disposable, configured D/G/prior update trials for startup stability.

The only proposed override is the owned generator optimizer group's LR. Every
trial is rolled back; selection preserves relative diversity/transmission on two
reserved input banks rather than maximizing a gradient norm or claiming quality.
"""
import copy
import json
import math
import time

import torch

from .checkpoints import capture_rng, data_contract, restore_rng, restore_trainer, trainer_state
from .initialization_tuning import _hash, _inventory, _storage, _structural
from .signal_diagnostic import _probe

TRIAL_STEPS = 8
MAX_TRIALS = 2
RETENTION = .25


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return copy.deepcopy(value)


def _snapshot(trainer):
    roots = ('graph', 'prior', 'ema_graph', 'ema_prior')
    gradients = []
    for name in roots:
        for parameter in getattr(trainer, name).parameters():
            gradients.append((parameter, parameter.grad,
                              None if parameter.grad is None else parameter.grad.detach().cpu().clone()))
    return {'state': _cpu_clone(trainer_state(trainer, None)), 'gradients': gradients,
            'metric_transfer': copy.deepcopy(trainer._metric_transfer),
            'unscale_scalars': copy.deepcopy(trainer._unscale_scalars)}


def _restore(trainer, snapshot):
    # The general checkpoint reader correctly rejects changed base rates; reset
    # this private trial override before asking it to restore the whole state.
    trainer.base_lrs = copy.deepcopy(snapshot['state']['base_lrs'])
    state = {**snapshot['state'], 'base_lrs': copy.deepcopy(snapshot['state']['base_lrs']),
             'data': copy.deepcopy(snapshot['state']['data'])}
    restore_trainer(trainer, state)
    trainer.base_lrs = copy.deepcopy(snapshot['state']['base_lrs'])
    for parameter, original, saved in snapshot['gradients']:
        if original is None:
            parameter.grad = None
        else:
            with torch.no_grad():
                original.copy_(saved)
            parameter.grad = original
    trainer._metric_transfer = copy.deepcopy(snapshot['metric_transfer'])
    trainer._unscale_scalars = copy.deepcopy(snapshot['unscale_scalars'])



def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right))
    return left == right


def _eligibility(trainer):
    from hndl.operators.pretrained import Pretrained
    if trainer.step != 0 or trainer.opt_g.state or trainer.opt_d.state:
        raise ValueError('Startup dynamics tuning requires step zero and empty optimizer states')
    if trainer.config['training']['steps'] < TRIAL_STEPS:
        return 'Configured training horizon is shorter than the 8-step diagnostic trial'
    contract = data_contract(trainer.data, trainer.config['data'])
    if not contract['supported']:
        return 'Data does not declare restorable or stateless behavior'
    if any(spec.get('factory') != 'hndl' for spec in trainer.config['components'].values()):
        return 'Dynamics trials currently require native HNDL ownership for every component'
    if any(spec['factory'] not in ('mse', 'l1') for spec in trainer.config['objectives']):
        return 'Custom objective hidden state cannot be safely rolled back'
    pretrained_paths = []
    for name, module in trainer.graph.named_modules():
        if isinstance(module, Pretrained):
            if any(parameter.requires_grad for parameter in module.parameters()):
                return 'Trainable pretrained parameters prohibit disposable optimizer trials'
            pretrained_paths.append(name)
    for name, module in trainer.graph.named_modules():
        if any(name == path or name.startswith(path + '.') for path in pretrained_paths):
            continue
        if not type(module).__module__.startswith(('torch.nn.', 'hndl.', 'hypergan.hndl_networks', 'hypergan.recipes')):
            return 'Unknown component module hidden state cannot be safely rolled back: ' + name
    layers, reason = _inventory(trainer)
    if not layers:
        return reason or 'No safely owned generator affine layers are available for the signal guard'
    protected_storages = {_storage(value) for _, value in _protected(trainer)}
    if any(parameter.requires_grad and _storage(parameter) in protected_storages
           for optimizer in (trainer.opt_g, trainer.opt_d) for group in optimizer.param_groups
           for parameter in group['params']):
        return 'Trainable optimizer tensors alias frozen/pretrained storage; disposable trials are prohibited'
    owned = {id(p) for p in trainer.program.generator_parameters}
    generator = {id(p) for p in trainer.graph.models['generator'].parameters() if p.requires_grad}
    group = {id(p) for p in trainer.opt_g.param_groups[0]['params']}
    if group != generator or group != owned:
        return 'The generator optimizer group also contains auxiliary or uncertain ownership'
    return None


def _protected(trainer):
    from hndl.operators.pretrained import Pretrained
    tensors, seen = [], set()
    for prefix, root in (('graph', trainer.graph), ('prior', trainer.prior)):
        for name, parameter in root.named_parameters():
            if not parameter.requires_grad:
                tensors.append((prefix + '.parameter.' + name, parameter))
                seen.add(id(parameter))
        for name, module in root.named_modules():
            params = list(module.parameters())
            frozen = params and all(not p.requires_grad for p in params)
            if isinstance(module, Pretrained) or frozen:
                for local, buffer in module.named_buffers():
                    if id(buffer) not in seen:
                        tensors.append((prefix + '.buffer.' + name + '.' + local, buffer))
                        seen.add(id(buffer))
    return tensors


def _first_gain(structural):
    rows = structural['affine_layers']
    return rows[0]['relative_cotangent_gain'] if rows else None


def _measure(trainer, banks, rng, protected, protected_hash):
    layers, _ = _inventory(trainer)
    measurements = []
    for batch, latent in banks:
        buffers = [(value, value.detach().clone()) for name in ('graph', 'prior')
                   for value in getattr(trainer, name).buffers()]
        try:
            restore_rng(rng)
            structural = _structural(trainer, batch, latent, layers)
            if _hash(protected) != protected_hash:
                raise ValueError('Dynamics measurement changed protected frozen/pretrained state')
        finally:
            with torch.no_grad():
                for value, saved in buffers:
                    value.copy_(saved)
        restore_rng(rng)
        signal = _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
        if _hash(protected) != protected_hash:
            raise ValueError('Dynamics measurement changed protected frozen/pretrained state')
        # Retain full scalar layer/parameter reports for interpretation, never graphs.
        measurements.append({'transmission': structural, 'signal': signal,
                             'first_relative_cotangent_gain': _first_gain(structural)})
    return measurements


def _guards(before, after):
    reasons, comparisons = [], []
    for index, (initial, final) in enumerate(zip(before, after)):
        row = {'bank': index}
        for label, old, new in (
            ('sample_diversity', initial['transmission']['sample_diversity_rms'], final['transmission']['sample_diversity_rms']),
            ('first_relative_cotangent_gain', initial['first_relative_cotangent_gain'], final['first_relative_cotangent_gain'])):
            valid = old is not None and new is not None and math.isfinite(old) and math.isfinite(new) and old > 1e-12
            retention = new / old if valid else None
            row[label + '_retention'] = retention
            if retention is None or retention < RETENTION:
                reasons.append(f'bank_{index}_{label}_retention_below_{RETENTION}')
        signal = final['signal']
        q = signal['summary']['generated_output_gradient_rms']
        generator_rows = [p for p in signal['parameters'] if p['path'].startswith('models.generator.') and p['status'] != 'frozen']
        if q is None or not math.isfinite(q) or q <= 0 or not any(p['status'] == 'measured' for p in generator_rows):
            reasons.append(f'bank_{index}_missing_finite_generator_signal')
        if any(p['status'] == 'nonfinite' for p in signal['parameters']):
            reasons.append(f'bank_{index}_nonfinite_parameter_gradient')
        if final['transmission']['nonfinite_output_fraction'] or signal['summary']['nonfinite_activation_records']:
            reasons.append(f'bank_{index}_nonfinite_activation_or_gradient')
        if not math.isfinite(signal['loss']):
            reasons.append(f'bank_{index}_nonfinite_loss')
        comparisons.append(row)
    return reasons, comparisons


def tune_startup_dynamics(trainer, *, progress=None):
    """Return a proposed G LR factor, leaving the trainer exactly at its input state.

    The baseline really runs first. Two data banks are reserved from its next
    draws; matching latent draws use the initial prior and the post-trial stream
    position. Dataset identities may repeat under sampling with replacement.
    Every rate candidate starts from the same original trainer and RNG state.
    """
    started = time.monotonic()
    progress = progress or (lambda row: None)
    exclusion = _eligibility(trainer)
    result = {'schema_version': 1, 'kind': 'startup-dynamics', 'trial_steps': TRIAL_STEPS,
              'maximum_disposable_updates': MAX_TRIALS * TRIAL_STEPS,
              'selected_g_lr_factor': 1.0, 'outcome': 'skipped' if exclusion else 'unresolved',
              'selected_candidate': None, 'reason': exclusion,
              'rate_formula': 'clip(minimum_retention, 0.1, 0.5), only after finite nonnegative baseline retention falls below 0.25',
              'candidates': [], 'guards': {'minimum_diversity_retention': RETENTION,
                                          'minimum_first_cotangent_gain_retention': RETENTION,
                                          'finite_nonzero_generator_objective_signal': True},
              'interpretation': ['Eight configured D/G/prior updates measure baseline drift; at most one derived-rate confirmation repeats those eight updates.',
                                 'Only the owned generator optimizer learning rate is proposed; discriminator/prior rates stay configured.',
                                 'Two matched banks are reserved after the baseline trial draw positions, with initial-prior latent tensors held fixed.',
                                 'Retention thresholds are startup-collapse heuristics, not image quality or a universal optimum.',
                                 'This finite-horizon check cannot certify long-term stability or detect every form of collapse.']}
    if exclusion:
        result['elapsed_seconds'] = time.monotonic() - started
        result['disposable_completed_updates'] = 0
        return result
    initial = _snapshot(trainer)
    protected = _protected(trainer)
    protected_hash = _hash(protected)
    result['protected_state_verification'] = {'before_sha256': protected_hash}
    completed_updates = 0

    def run_trial(factor, index):
        nonlocal completed_updates
        trainer.base_lrs[0][0] = initial['state']['base_lrs'][0][0] * factor
        trainer.opt_g.param_groups[0]['lr'] = initial['state']['optimizers'][0]['param_groups'][0]['lr'] * factor
        losses = []
        for step in range(1, TRIAL_STEPS + 1):
            progress({'phase': 'dynamics', 'candidate': index, 'total_candidates': MAX_TRIALS,
                      'trial_step': step, 'trial_steps': TRIAL_STEPS, 'lr_factor': factor,
                      'message': f'Testing generator LR ×{factor:g}: update {step}/{TRIAL_STEPS}'})
            row, _ = trainer.update()
            completed_updates += 1
            if _hash(protected) != protected_hash:
                raise ValueError('Dynamics trial changed protected frozen/pretrained state')
            losses.append({key: float(row[key]) for key in ('g_loss', 'd_loss')})
            if any(not math.isfinite(value) for value in losses[-1].values()):
                raise FloatingPointError('Nonfinite loss during disposable dynamics trial')
        return losses

    try:
        try:
            baseline_losses = run_trial(1.0, 1)
        except (FloatingPointError, ValueError) as error:
            # A partial D/G failure has no common full-horizon tail input bank.
            # Refuse to invent one or declare that shrinking G LR fixes D/prior.
            if 'nonfinite' not in str(error).lower():
                raise
            result.update(reason='Baseline trial became nonfinite before a full held-out comparison: ' + str(error))
            result['candidates'].append({'lr_factor': 1., 'accepted': False, 'failure': str(error)})
            return result
        baseline_final = _snapshot(trainer)
        tail_prior_state = trainer.streams['prior'].get_state().clone()
        real_banks = [trainer.batch(), trainer.batch()]
        _restore(trainer, initial)
        local_prior = torch.Generator(device=trainer.device)
        local_prior.set_state(tail_prior_state)
        with torch.no_grad():
            banks = [(batch, trainer.prior.sample(len(batch['real']), generator=local_prior)) for batch in real_banks]
        probe_rng = capture_rng()
        before = _measure(trainer, banks, probe_rng, protected, protected_hash)
        result['before'] = before
        _restore(trainer, baseline_final)
        after = _measure(trainer, banks, probe_rng, protected, protected_hash)
        reasons, comparisons = _guards(before, after)
        result['candidates'].append({'name': 'g_lr_1', 'lr_factor': 1., 'accepted': not reasons,
                                     'rejection_reasons': reasons, 'comparisons': comparisons,
                                     'after': after, 'trial_losses': baseline_losses})
        del baseline_final
        if not reasons:
            result.update(outcome='kept_baseline', selected_candidate='g_lr_1',
                          reason='Configured generator learning rate passed both reserved-bank startup retention guards')
            return result
        retentions = [value for row in comparisons for key, value in row.items() if key.endswith('_retention')]
        valid = retentions and all(value is not None and math.isfinite(value) and value >= 0 for value in retentions)
        if not valid or min(retentions) >= RETENTION:
            result['reason'] = 'Startup guards failed without finite nonnegative retention drift from which to derive a rate correction'
            return result
        minimum_retention = min(retentions)
        factor = max(.1, min(.5, minimum_retention))
        result['derived_rate'] = {'minimum_retention': minimum_retention, 'proposed_g_lr_factor': factor}
        _restore(trainer, initial)
        try:
            losses = run_trial(factor, 2)
            after = _measure(trainer, banks, probe_rng, protected, protected_hash)
            reasons, comparisons = _guards(before, after)
        except (FloatingPointError, ValueError) as error:
            if 'nonfinite' not in str(error).lower():
                raise
            result['candidates'].append({'name': 'derived_g_lr', 'lr_factor': factor,
                                         'accepted': False, 'failure': str(error)})
            result['reason'] = 'Derived-rate confirmation became nonfinite; baseline rate remains unchanged'
            return result
        result['candidates'].append({'name': 'derived_g_lr', 'lr_factor': factor,
                                     'accepted': not reasons, 'rejection_reasons': reasons,
                                     'comparisons': comparisons, 'after': after, 'trial_losses': losses})
        if not reasons:
            result.update(outcome='selected', selected_g_lr_factor=factor, selected_candidate='derived_g_lr',
                          reason='One drift-derived smaller generator learning rate passed both reserved-bank startup retention guards')
            return result
        result['reason'] = 'The single derived-rate confirmation failed startup retention guards; baseline rate remains unchanged'
        return result
    finally:
        protected_after = _hash(protected)
        _restore(trainer, initial)
        result['disposable_completed_updates'] = completed_updates
        result['restored_step'] = trainer.step
        result['protected_state_verification'].update(after_sha256=protected_after,
                                                      unchanged_during_trials=protected_after == protected_hash,
                                                      after_restore_sha256=_hash(protected))
        if not _same_state(initial['state'], trainer_state(trainer, None)):
            raise ValueError('Disposable dynamics trial did not restore exact trainer state')
        if any(parameter.grad is not original or (saved is not None and not _same_state(saved, parameter.grad))
               for parameter, original, saved in initial['gradients']):
            raise ValueError('Disposable dynamics trial did not restore existing gradient fields')
        result['all_trial_state_restored'] = True
        result['elapsed_seconds'] = time.monotonic() - started
        json.dumps(result, allow_nan=False)
