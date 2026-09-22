"""Measured optimizer-direction proposals, verified by one disposable coupled replay."""
import copy
import json
import math
import time

import torch

from .checkpoints import capture_rng, data_contract, restore_rng, restore_trainer, trainer_state
from .signal_structure import _hash, _inventory, _storage, _structural
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
            'unscale_scalars': copy.deepcopy(trainer._unscale_scalars),
            'warmup_present': hasattr(trainer, 'g_lr_warmup'),
            'warmup': copy.deepcopy(getattr(trainer, 'g_lr_warmup', None)),
            'compiled_caches': [(module, module._compiled) for name in roots
                                for module in getattr(trainer, name).modules()
                                if type(module).__module__ == 'hndl.torch' and hasattr(module, '_compiled')]}


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
    if snapshot['warmup_present']:
        trainer.g_lr_warmup = copy.deepcopy(snapshot['warmup'])
    elif hasattr(trainer, 'g_lr_warmup'):
        del trainer.g_lr_warmup
    for module, compiled in snapshot['compiled_caches']:
        module._compiled = compiled



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
    if len(trainer.opt_d.param_groups) != 1:
        return 'Dynamics discriminator tuning requires one native critic optimizer group'
    critics = {id(p) for p in trainer.program.critic_parameters}
    critic_group = {id(p) for group in trainer.opt_d.param_groups for p in group['params']}
    declared_critics = {id(p) for term in trainer.program.adversarial_terms
                        for p in term.module.parameters() if p.requires_grad}
    if critics != critic_group or critics != declared_critics or critics & (owned | {id(p) for p in trainer.prior.parameters()}):
        return 'The discriminator optimizer contains uncertain or overlapping ownership'
    role_storages = [{_storage(parameter) for parameter in parameters} for parameters in
                     (trainer.program.generator_parameters, trainer.program.critic_parameters,
                      trainer.program.prior_parameters)]
    if any(first & second for index, first in enumerate(role_storages)
           for second in role_storages[index + 1:]):
        return 'Generator, discriminator and prior optimizer groups share tensor storage; fixed-player probes are prohibited'
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



def _optimizer_motion(trainer, initial):
    """Actual parameter displacement, including Adam/moments and configured LR.

    Positive displacement only proves an optimizer moved a player, not that the
    update was useful. No candidate is ranked by displacement or gradient size.
    """
    originals = {}
    for root_name in ('graph', 'prior'):
        root = getattr(trainer, root_name)
        for name, parameter in root.named_parameters():
            originals[id(parameter)] = initial['state'][root_name][name]
    roles = {'generator': trainer.program.generator_parameters,
             'discriminator': trainer.program.critic_parameters,
             'prior': trainer.program.prior_parameters}
    report = {}
    for role, parameters in roles.items():
        squared_delta, squared_initial, elements, finite = 0., 0., 0, True
        for parameter in parameters:
            current = parameter.detach().cpu().to(torch.float64)
            old = originals[id(parameter)].to(torch.float64)
            elements += current.numel()
            finite = finite and bool(torch.isfinite(current).all())
            squared_delta += float((current - old).square().sum())
            squared_initial += float(old.square().sum())
        valid = finite and math.isfinite(squared_delta) and math.isfinite(squared_initial)
        report[role] = {'elements': elements, 'finite': valid,
                        'changed': valid and squared_delta > 0,
                        'delta_rms': math.sqrt(squared_delta / elements) if valid and elements else None,
                        'measurement': 'cumulative_eight_update_parameter_displacement',
                        'initial_rms': math.sqrt(squared_initial / elements) if valid and elements else None,
                        'relative_delta_l2': math.sqrt(squared_delta / squared_initial)
                            if valid and squared_initial > 0 else None}
    return report


def _candidate_guards(before, after, motion):
    reasons, comparisons = _guards(before, after)
    for role in ('generator', 'discriminator'):
        if not motion[role]['finite']:
            reasons.append(role + '_optimizer_displacement_nonfinite')
        elif not motion[role]['changed']:
            reasons.append(role + '_optimizer_did_not_move')
    return reasons, comparisons

def tune_startup_dynamics(trainer, *, progress=None):
    """Fit the first G and eighth D directions; verify one paired replay.

    All optimizer trials, objective probes and reserved draws are discarded.
    Phase-local objectives include the configured penalty schedule and auxiliary
    terms. Prior rates and pretrained state are never calibration variables.
    """
    from .startup_response_probe import UpdateObserver, PhaseProbes
    from .gradient_response import aggregate_gradient_response, differentiation_interval, fit_gradient_response
    from .update_response import verify_player_validation
    started = time.monotonic()
    progress = progress or (lambda row: None)
    exclusion = _eligibility(trainer)
    budget = {'training_updates': 0, 'gradient_field_phase_loss_evaluations': 0,
              'gradient_field_player_gradient_evaluations': 0,
              'validation_phase_loss_evaluations': 0, 'g_response_forwards': 0,
              'q_image_forwards': 0, 'd_signal_input_backwards': 0,
              'guard_structural_evaluations': 0, 'guard_signal_evaluations': 0}
    result = {'schema_version': 2, 'kind': 'startup-dynamics', 'method': 'measured-update-response',
              'trial_steps': TRIAL_STEPS, 'maximum_disposable_updates': 16,
              'anchor_steps': {'generator': 1, 'discriminator': 8},
              'selected_g_lr_factor': 1., 'selected_d_lr_factor': 1.,
              'outcome': 'skipped' if exclusion else 'unresolved', 'selected_candidate': None,
              'reason': exclusion, 'candidates': [], 'probe_budget': budget,
              'rate_formula': 'min(1, -g0.dot(delta)/(2*C)); C=norm_M(delta)*norm_inverse_M(gh-g0)/h, with frozen post-update Adam denominator M',
              'proposal_method': 'adam-metric-gradient-response',
              'selection_policy': 'Two fit banks per player, two separate strict-decrease validation banks, then at most one coupled eight-update replay.',
              'guards': {'minimum_diversity_retention': RETENTION,
                         'minimum_first_cotangent_gain_retention': RETENTION,
                         'finite_nonzero_generator_objective_signal': True,
                         'finite_nonzero_generator_and_discriminator_optimizer_displacement': True},
              'interpretation': [
                  'G uses its first actual optimizer displacement before the startup transient; D uses update eight with the configured penalty schedule.',
                  'Fixed-latent G response isolates generator motion; learned-prior displacement is reported separately and its rate remains configured.',
                  'Matching-bank gradients at factors zero and h measure change in magnitude and direction; signed loss curvature need not be positive.',
                  'The observed secant and numerical differentiation interval are estimates, not certified smoothness bounds or universal ideal signal targets.',
                  'Each player is measured against its phase-local frozen opponent; only a coupled replay can test the proposed pair.',
                  'Input-gradient changes and retention guards describe local training behavior, not semantic quality or long-term stability.']}
    if exclusion:
        result.update(elapsed_seconds=time.monotonic() - started, disposable_completed_updates=0)
        return result
    initial = _snapshot(trainer)
    protected = _protected(trainer)
    protected_hash = _hash(protected)
    result['protected_state_verification'] = {'before_sha256': protected_hash}
    previous_observer = getattr(trainer, '_update_response_observer', None)
    observer_present = hasattr(trainer, '_update_response_observer')
    observer = UpdateObserver(trainer, _snapshot, protected, protected_hash, budget)
    probes = PhaseProbes(trainer, _restore, protected, protected_hash, budget)

    def event(stage, *, step=8, g_factor=1., d_factor=1., message):
        progress({'phase': 'dynamics', 'stage': stage, 'candidate': 2 if stage == 'replay' else 1,
                  'total_candidates': MAX_TRIALS, 'trial_step': step, 'trial_steps': TRIAL_STEPS,
                  'lr_factor': g_factor, 'g_lr_factor': g_factor, 'd_lr_factor': d_factor,
                  'message': message})

    def run_trial(g_factor, d_factor, *, stage, observer):
        trainer.base_lrs[0][0] = initial['state']['base_lrs'][0][0] * g_factor
        trainer.opt_g.param_groups[0]['lr'] = initial['state']['optimizers'][0]['param_groups'][0]['lr'] * g_factor
        trainer.base_lrs[1][0] = initial['state']['base_lrs'][1][0] * d_factor
        trainer.opt_d.param_groups[0]['lr'] = initial['state']['optimizers'][1]['param_groups'][0]['lr'] * d_factor
        trainer._update_response_observer = observer
        losses = []
        for step in range(1, TRIAL_STEPS + 1):
            event(stage, step=step, g_factor=g_factor, d_factor=d_factor,
                  message=f'{"Measuring configured updates" if stage == "measure" else "Verifying coupled proposal"}: {step}/{TRIAL_STEPS}')
            row, _ = trainer.update()
            budget['training_updates'] += 1
            if _hash(protected) != protected_hash:
                raise ValueError('Dynamics trial changed protected frozen/pretrained state')
            losses.append({key: float(row[key]) for key in ('g_loss', 'd_loss')})
            if any(not math.isfinite(value) for value in losses[-1].values()):
                raise FloatingPointError('Nonfinite loss during disposable dynamics trial')
        trainer._update_response_observer = None
        return losses

    def measure(banks, rng):
        budget['guard_structural_evaluations'] += len(banks)
        budget['guard_signal_evaluations'] += len(banks)
        return _measure(trainer, banks, rng, protected, protected_hash)

    try:
        baseline_losses = run_trial(1., 1., stage='measure', observer=observer)
        if set(observer.anchors) != {'generator', 'discriminator'}:
            raise ValueError('Configured update did not expose both phase-local optimizer anchors')
        baseline_motion = _optimizer_motion(trainer, initial)
        baseline_final = _snapshot(trainer)
        # Real draws are reserved after the baseline. Reuse the same tail
        # sampling RNG under each phase's prior, rather than fitting initial G
        # against latent values produced by an already updated learned prior.
        with torch.no_grad():
            batches = [_cpu_clone(trainer.batch()) for _ in range(4)]
            prior_rng = trainer.streams['prior'].get_state().clone()
            role_banks = {}
            for role in ('generator', 'discriminator'):
                _restore(trainer, observer.anchors[role]['snapshot'])
                trainer.streams['prior'].set_state(prior_rng)
                role_banks[role] = [(batch, _cpu_clone(trainer.prior.sample(
                    trainer.config['training']['batch_size'], generator=trainer.streams['prior'])))
                    for batch in batches]
                if _hash(protected) != protected_hash:
                    raise ValueError('Reserved prior draws changed protected frozen/pretrained state')
        validation_banks = role_banks['generator'][2:]
        probe_rng = capture_rng()
        result['bank_control'] = {'fit_draws': [0, 1], 'validation_draws': [2, 3],
                                  'origin': 'four reserved real draws after baseline; phase-local prior latent tensors from the same tail sampling RNG',
                                  'latent_prior_steps': {'generator': 0, 'discriminator': 7},
                                  'guard_latents': 'fixed generator-anchor prior values',
                                  'replacement_sampling_may_repeat_dataset_items': True}
        _restore(trainer, initial)
        before = measure(validation_banks, probe_rng)
        result['before'] = before
        _restore(trainer, baseline_final)
        baseline_after = measure(validation_banks, probe_rng)
        baseline_failures, comparisons = _candidate_guards(before, baseline_after, baseline_motion)
        result['candidates'].append({'name': 'baseline', 'lr_factor': 1., 'd_lr_factor': 1.,
                                     'accepted': not baseline_failures, 'rejection_reasons': baseline_failures,
                                     'comparisons': comparisons, 'after': baseline_after,
                                     'trial_losses': baseline_losses, 'optimizer_motion': baseline_motion,
                                     'update_response': observer.observations})
        event('fit', message='Measuring first-G and eighth-D gradient response in their Adam metrics')
        proposals = {}
        for role in ('generator', 'discriminator'):
            anchor = observer.anchors[role]
            motion = anchor['optimizer_motion']
            size = math.sqrt(motion['elements'])
            interval = differentiation_interval(motion['initial_rms'] * size, motion['delta_rms'] * size)
            fits, measurements = [], []
            for bank in role_banks[role][:2]:
                if interval['status'] == 'valid':
                    measurement = probes.gradient_change(anchor, role, bank, interval['h'],
                                                         adam_denominators=anchor['adam_denominators'])
                else:
                    measurement = {'status': 'unresolved', 'reason': interval['reason']}
                measurements.append(measurement)
                fits.append(fit_gradient_response(measurement))
            proposals[role] = {**aggregate_gradient_response(fits), 'banks': fits,
                               'measurements': measurements, 'differentiation_interval': interval}
        result['directional_proposals'] = proposals
        result['d_signal_response'] = probes.d_signal_response(observer.anchors['discriminator'], role_banks['discriminator'][:2])
        g_factor, d_factor = proposals['generator']['factor'], proposals['discriminator']['factor']
        if g_factor == d_factor == 1.:
            if all(proposal['status'] == 'unchanged' for proposal in proposals.values()) and not baseline_failures:
                result.update(outcome='kept_baseline', selected_candidate='baseline',
                              reason='Both gradient-response estimates resolved no reduction and the baseline passed startup guards')
            else:
                result['reason'] = 'Neither player supplied a resolved reduction across both fitting banks; configured rates retained'
            return result
        event('validate', g_factor=g_factor, d_factor=d_factor,
              message='Checking the proposed pair on two separate validation banks')
        validations = {}
        for role, factor in (('generator', g_factor), ('discriminator', d_factor)):
            losses, epsilon = [], torch.finfo(torch.float32).eps
            if factor != 1.:
                for bank in role_banks[role][2:]:
                    pair = [probes.evaluate(observer.anchors[role], role, bank, point,
                            category='validation_phase_loss_evaluations') for point in (0., factor)]
                    losses.append(tuple(value for value, _ in pair))
                    epsilon = max(epsilon, *(precision for _, precision in pair))
            validations[role] = verify_player_validation(losses, changed=factor != 1., epsilon=epsilon)
        result['heldout_validation'] = validations
        if not all(value['accepted'] for value in validations.values()):
            result['reason'] = 'A changed player failed strict held-out decrease; the whole pair was rejected without another proposal'
            return result
        _restore(trainer, initial)
        replay_observer = UpdateObserver(trainer, _snapshot, protected, protected_hash, budget,
                                         capture_anchors=False)
        losses = run_trial(g_factor, d_factor, stage='replay', observer=replay_observer)
        motion = _optimizer_motion(trainer, initial)
        measured = measure(validation_banks, probe_rng)
        failures, comparisons = _candidate_guards(before, measured, motion)
        result['candidates'].append({'name': 'measured_update_pair', 'lr_factor': g_factor,
                                     'd_lr_factor': d_factor, 'accepted': not failures,
                                     'rejection_reasons': failures, 'comparisons': comparisons,
                                     'after': measured, 'trial_losses': losses, 'optimizer_motion': motion,
                                     'update_response': replay_observer.observations})
        if failures:
            result['reason'] = 'The one coupled replay failed startup guards; the whole pair was rejected'
        else:
            result.update(outcome='selected', selected_g_lr_factor=g_factor, selected_d_lr_factor=d_factor,
                          selected_candidate='measured_update_pair',
                          reason='The measured pair passed separate held-out loss checks and one coupled replay')
        return result
    except (FloatingPointError, ValueError) as error:
        if not isinstance(error, FloatingPointError) and not str(error).startswith('Nonfinite '):
            raise
        result['reason'] = 'Nonfinite disposable measurement or update; configured rates retained: ' + str(error)
        return result
    finally:
        protected_after = _hash(protected)
        _restore(trainer, initial)
        if observer_present:
            trainer._update_response_observer = previous_observer
        elif hasattr(trainer, '_update_response_observer'):
            del trainer._update_response_observer
        result['disposable_completed_updates'] = budget['training_updates']
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
