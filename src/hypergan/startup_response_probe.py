"""Private phase-local measurements for the disposable startup update trial."""
import math

import torch

from .checkpoints import capture_rng, restore_rng
from .objective_program import _bound_scores, _generator_tail, _sum_tensors, score_candidate
from .signal_structure import _hash, _inventory
from .update_response import tensor_change


def _parameters(trainer, role):
    return {'generator': trainer.program.generator_parameters,
            'discriminator': trainer.program.critic_parameters,
            'prior': trainer.program.prior_parameters}[role]


def _values(parameters):
    return [value.detach().cpu().clone() for value in parameters]


def _registered_parameters(trainer):
    return [(prefix + '.' + name, parameter) for prefix, root in
            (('graph', trainer.graph), ('prior', trainer.prior)) for name, parameter in root.named_parameters()]


def _copy(parameters, values):
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            parameter.copy_(value)


def capture_adam_denominators(trainer, role, *, observation_step=None):
    """Capture sqrt(v_hat)+eps from the current, already completed Adam step.

    CPU tensors in this private descriptor are inputs to ``gradient_change``,
    not a JSON artifact. No optimizer state is initialized or changed here.
    """
    from .training import DeviceAdam
    if role not in ('generator', 'discriminator'):
        raise ValueError('Adam metric requires generator or discriminator ownership')
    if observation_step is not None and (type(observation_step) is not int or observation_step < 1):
        raise ValueError('Adam metric observation step must be a positive integer')
    optimizer = trainer.opt_g if role == 'generator' else trainer.opt_d
    report = {'status': 'unsupported', 'reason': None, 'kind': 'bias_corrected_adam_denominator',
              'player': role, 'parameter_paths': [], 'steps': [], 'denominators': []}
    if observation_step is not None:
        report['observation_step'] = observation_step
    if type(optimizer) not in (torch.optim.Adam, DeviceAdam):
        report['reason'] = 'Only native Adam and DeviceAdam metrics are supported'
        return report
    groups = {id(parameter): group for group in optimizer.param_groups for parameter in group['params']}
    paths = {id(parameter): path for path, parameter in _registered_parameters(trainer)}
    denominators, names, steps = [], [], []
    for parameter in _parameters(trainer, role):
        group, state = groups.get(id(parameter)), optimizer.state.get(parameter)
        if group is None or group.get('amsgrad', False) or not state or 'exp_avg_sq' not in state or 'step' not in state:
            report['reason'] = 'Every owned parameter needs completed, non-AMSGrad Adam second-moment state'
            return report
        step = float(state['step'])
        beta2, epsilon = group['betas'][1], group['eps']
        if (not math.isfinite(step) or step < 1 or not step.is_integer()
                or not 0 <= beta2 < 1 or not math.isfinite(epsilon) or epsilon <= 0):
            report['reason'] = 'Invalid Adam step, beta2 or epsilon'
            return report
        moment = state['exp_avg_sq'].detach().cpu().double()
        if moment.shape != parameter.shape or not bool(torch.isfinite(moment).all()) or bool((moment < 0).any()):
            report['reason'] = 'Invalid Adam second-moment tensor'
            return report
        denominator = (moment / (1. - beta2 ** step)).sqrt().add_(epsilon)
        if not bool(torch.isfinite(denominator).all()):
            report['reason'] = 'Unresolved Adam denominator'
            return report
        denominators.append(denominator)
        names.append(paths[id(parameter)])
        steps.append(int(step))
    report.update(status='measured', parameter_paths=names, steps=steps, denominators=denominators)
    return report


def displacement(parameters, before):
    deltas, squared, initial_squared, slope, elements = [], 0., 0., 0., 0
    for parameter, old in zip(parameters, before):
        delta = parameter.detach().cpu() - old
        deltas.append(delta)
        squared += float(delta.double().square().sum())
        initial_squared += float(old.double().square().sum())
        if parameter.grad is not None:
            slope += float((parameter.grad.detach().cpu().double() * delta.double()).sum())
        elements += delta.numel()
    finite = all(math.isfinite(value) for value in (squared, initial_squared, slope))
    return deltas, {'elements': elements, 'finite': finite, 'changed': finite and squared > 0,
                    'delta_rms': math.sqrt(squared / elements) if finite and elements else None,
                    'initial_rms': math.sqrt(initial_squared / elements) if finite and elements else None,
                    'relative_delta_l2': math.sqrt(squared / initial_squared) if finite and initial_squared > 0 else None,
                    'training_gradient_dot_delta': slope if finite else None,
                    'measurement': 'one_actual_optimizer_update'}


def _ephemeral(trainer):
    return {'rng': capture_rng(),
            'streams': {name: stream.get_state().clone() for name, stream in trainer.streams.items()},
            'buffers': [(value, value.detach().clone()) for root in (trainer.graph, trainer.prior)
                        for value in root.buffers()],
            'modes': [(module, module.training) for root in (trainer.graph, trainer.prior)
                      for module in root.modules()],
            'compiled_caches': [(module, module._compiled) for root in (trainer.graph, trainer.prior)
                                for module in root.modules()
                                if type(module).__module__ == 'hndl.torch' and hasattr(module, '_compiled')]}


def _restore_ephemeral(trainer, saved):
    with torch.no_grad():
        for value, original in saved['buffers']:
            value.copy_(original)
    for module, mode in saved['modes']:
        module.training = mode
    for module, compiled in saved['compiled_caches']:
        module._compiled = compiled
    for name, value in saved['streams'].items():
        trainer.streams[name].set_state(value)
    restore_rng(saved['rng'])


def _image_response(before, after):
    result = {'full': tensor_change(before, after)}
    if before.ndim == 4:
        result['per_image_mean_removed'] = tensor_change(
            before - before.mean((-2, -1), keepdim=True), after - after.mean((-2, -1), keepdim=True))
        result['spatial_channel_means'] = tensor_change(before.mean((-2, -1)), after.mean((-2, -1)))
        if min(before.shape[-2:]) >= 4:
            result['pooled_4x4'] = tensor_change(torch.nn.functional.adaptive_avg_pool2d(before, (4, 4)),
                                                 torch.nn.functional.adaptive_avg_pool2d(after, (4, 4)))
    return result


def _final_owned_affine(trainer):
    """Identify an unambiguous output affine through layout changes and tanh.

    Registration order alone does not establish a final layer in a branched
    graph. Follow the native output dependency instead, and abstain on unknown
    transformations or ownership rather than labeling an arbitrary activation.
    """
    generator = trainer.graph.models['generator']
    network = getattr(generator, 'network', None)
    plan = getattr(network, 'plan', None)
    if plan is None or not isinstance(getattr(plan, 'output_ref', None), str):
        return None, {'status': 'skipped', 'reason': 'No single native HNDL output dependency is available'}
    owned, reason = _inventory(trainer)
    by_module = {id(module): name for name, module in owned}
    nodes = {f'node:{node.id}/{port}': node for node in plan.nodes for port in node.outputs}
    reference, downstream, visited = plan.output_ref, [], set()
    while reference in nodes and reference not in visited:
        visited.add(reference)
        node = nodes[reference]
        module = network.nodes['n_' + node.id]
        if id(module) in by_module:
            return module, {'status': 'measured', 'path': 'generator.' + by_module[id(module)],
                            'measurement': 'final_owned_affine_activation_displacement',
                            'relationship_to_output': 'pre_tanh' if 'tanh' in downstream else 'output_up_to_layout',
                            'downstream_operations': list(reversed(downstream)),
                            'interpretation': 'Matched fixed-latent observation only; no threshold or tuning decision uses this displacement'}
        operation = node.op.split('@', 1)[0]
        if (operation not in ('reshape', 'permute', 'transpose', 'flatten', 'identity', 'contiguous', 'tanh')
                or len(node.inputs) != 1 or len(node.outputs) != 1 or (operation == 'tanh' and 'tanh' in downstream)):
            return None, {'status': 'skipped', 'reason': reason or 'Output path has no unambiguous owned affine through supported layout operations and optional tanh'}
        downstream.append(operation)
        reference = next(iter(node.inputs.values()))
    return None, {'status': 'skipped', 'reason': reason or 'Output dependency does not reach a safely owned affine layer'}


def _activation_scalars(value, *, pre_tanh=False):
    """Descriptive scalar observations, never acceptance thresholds."""
    value = value.detach().to(device='cpu', dtype=torch.float64)
    finite = bool(torch.isfinite(value).all())
    report = {'status': 'finite' if finite else 'nonfinite', 'shape': list(value.shape),
              'rms': None, 'std': None, 'absolute_max': None, 'absolute_above_0_99_fraction': None}
    if not finite or not value.numel():
        return report
    report.update(rms=float(value.square().mean().sqrt()), std=float(value.std(unbiased=False)),
                  absolute_max=float(value.abs().max()),
                  absolute_above_0_99_fraction=float((value.abs() > .99).double().mean()))
    if pre_tanh:
        derivative = 1. - value.tanh().square()
        report['tanh_response'] = {'mean_derivative': float(derivative.mean()),
                                   'minimum_derivative': float(derivative.min()),
                                   'derivative_below_0_01_fraction': float((derivative < .01).double().mean()),
                                   'interpretation': 'Descriptive tanh saturation; no tuning threshold is applied'}
    return report


class UpdateObserver:
    """Observe updates one/eight; anchor first G and eighth D, never graphs."""
    def __init__(self, trainer, snapshot, protected, protected_hash, budget, *, capture_anchors=True):
        self.trainer, self.snapshot = trainer, snapshot
        self.protected, self.protected_hash, self.budget = protected, protected_hash, budget
        self.anchors, self.observations, self.pending = {}, [], {}
        self.capture_anchors = capture_anchors
        self.affine_module, self.affine_description = _final_owned_affine(trainer)

    def __call__(self, event, *, step, batch, ids, context):
        if step not in (1, 8):
            return
        role = 'discriminator' if event.endswith('_d') else 'generator'
        anchor_step = 1 if role == 'generator' else 8
        parameters = _parameters(self.trainer, role)
        if event.startswith('before'):
            entry = {'before': _values(parameters)}
            if role == 'generator':
                entry['prior_before'] = _values(_parameters(self.trainer, 'prior'))
                entry['latent'] = context['latent'].detach().clone()
            if step == anchor_step and self.capture_anchors:
                entry['snapshot'] = self.snapshot(self.trainer)
            self.pending[role] = entry
            return
        entry = self.pending.pop(role)
        deltas, motion = displacement(parameters, entry['before'])
        row = {'step': step, 'player': role, 'optimizer_motion': motion}
        if step == anchor_step and self.capture_anchors:
            self.anchors[role] = {'snapshot': entry['snapshot'], 'before': entry['before'],
                                  'delta': deltas, 'step': step, 'optimizer_motion': dict(motion),
                                  'adam_denominators': capture_adam_denominators(
                                      self.trainer, role, observation_step=step)}
        if role == 'generator':
            _, row['prior_optimizer_motion'] = displacement(_parameters(self.trainer, 'prior'), entry['prior_before'])
            # Two replay forwards avoid assuming the original stochastic forward
            # can be reused. Fixed latent values deliberately exclude prior motion.
            after = _values(parameters)
            fence = _ephemeral(self.trainer)
            affine_values = []
            handle = None
            try:
                if self.affine_module is not None:
                    def observe_affine(module, args, output):
                        affine_values.append(output.detach().cpu().clone() if isinstance(output, torch.Tensor) else None)
                    handle = self.affine_module.register_forward_hook(observe_affine)
                outputs = []
                affine_pairs = []
                for values in (entry['before'], after):
                    affine_values.clear()
                    _copy(parameters, values)
                    _restore_ephemeral(self.trainer, fence)
                    expected = _hash(_registered_parameters(self.trainer))
                    with torch.no_grad():
                        output = self.trainer.graph.generate(entry['latent'], batch, prior=self.trainer.prior)['generated']
                    self.budget['g_response_forwards'] += 1
                    if _hash(_registered_parameters(self.trainer)) != expected:
                        raise ValueError('Update response changed registered parameters')
                    if _hash(self.protected) != self.protected_hash:
                        raise ValueError('Update response changed protected frozen/pretrained state')
                    outputs.append(output.detach().cpu())
                    affine_pairs.append(affine_values[0] if len(affine_values) == 1 else None)
                row['generator_output_response'] = _image_response(*outputs)
                affine_report = dict(self.affine_description)
                if self.affine_module is not None:
                    if any(value is None for value in affine_pairs) or affine_pairs[0].shape != affine_pairs[1].shape:
                        affine_report.update(status='skipped', reason='Final owned affine was not invoked exactly once with matching tensor outputs')
                    else:
                        affine_report.update(shape=list(affine_pairs[0].shape),
                                             response=tensor_change(*affine_pairs))
                row['generator_final_affine_response'] = affine_report
                row['input_control'] = 'fixed_latent_values; prior output response not included'
            finally:
                if handle is not None:
                    handle.remove()
                _copy(parameters, after)
                _restore_ephemeral(self.trainer, fence)
        self.observations.append(row)


def phase_loss(trainer, role, batch, latent, *, step):
    """Complete configured phase objective, without an optimizer operation."""
    batch, ids, context = trainer._draw(batch, latent)
    weighted, penalties, fake = [], [], None
    critic = role == 'discriminator'
    for term in trainer.program.adversarial_terms:
        phase = term.critic_phase if critic else term.generator_phase
        real, fake, real_score, fake_score = _bound_scores(
            term, context, trainer.graph, 'critic' if critic else 'generator', phase,
            first='real' if critic else 'fake')
        weighted.append(term.weight * (term.gan.d_loss(real_score, fake_score) if critic
                                       else term.gan.g_loss(fake_score, real_score)))
        if critic and term.penalty_fn is not None:
            penalties.append(term.penalty_fn(
                lambda value, term=term: score_candidate(term, value, context, trainer.graph, 'critic'),
                real, fake, step=step, generator=trainer.streams['penalty']))
    loss = _sum_tensors(weighted)
    if critic:
        loss = loss + _sum_tensors(penalties) if penalties else loss
    else:
        prior_loss, objectives = _generator_tail(trainer, trainer.program, context, ids, fake)
        loss = loss + prior_loss + sum(objectives)
    return loss


class PhaseProbes:
    def __init__(self, trainer, restore, protected, protected_hash, budget):
        self.trainer, self.restore = trainer, restore
        self.protected, self.protected_hash, self.budget = protected, protected_hash, budget

    def evaluate(self, anchor, role, bank, factor, *, category):
        trainer = self.trainer
        self.restore(trainer, anchor['snapshot'])
        try:
            _copy(_parameters(trainer, role), [before + factor * delta
                                              for before, delta in zip(anchor['before'], anchor['delta'])])
            parameters = _registered_parameters(trainer)
            expected = _hash(parameters)
            self.budget[category] += 1
            loss = phase_loss(trainer, role, *bank, step=anchor['step'])
            value, epsilon = float(loss.detach()), torch.finfo(loss.dtype).eps
            if _hash(parameters) != expected:
                raise ValueError('Phase loss measurement changed registered parameters')
            if _hash(self.protected) != self.protected_hash:
                raise ValueError('Phase loss measurement changed protected frozen/pretrained state')
            return value, epsilon
        finally:
            self.restore(trainer, anchor['snapshot'])

    def measure_direction(self, anchor, role, bank, factor, *, gradient=False):
        """Read-only loss/activation/optional exact directional-derivative audit.

        Every point uses the anchor's same opponent, prior, buffers and random
        draws. The derivative is evaluated on this bank at this factor, along
        the signed actual optimizer displacement. No parameter ``.grad`` is
        populated and no optimizer runs. Unlike ``evaluate``, this diagnostic
        restores its caller's complete trainer state, even on failure.
        """
        from .startup_dynamics import _snapshot
        if role not in ('generator', 'discriminator'):
            raise ValueError('Direction audit requires generator or discriminator ownership')
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor < 0:
            raise ValueError('Direction audit factor must be finite and nonnegative')
        trainer = self.trainer
        entry = _snapshot(trainer)
        handles = []
        try:
            self.restore(trainer, anchor['snapshot'])
            owned = _parameters(trainer, role)
            _copy(owned, [before + factor * delta for before, delta in zip(anchor['before'], anchor['delta'])])
            parameters = _registered_parameters(trainer)
            expected = _hash(parameters)
            outputs, affine_outputs = [], []
            affine, description = _final_owned_affine(trainer)
            def capture(target):
                def observe(module, args, output):
                    target.append(output.detach().cpu().clone() if isinstance(output, torch.Tensor) else None)
                return observe
            handles.append(trainer.graph.models['generator'].register_forward_hook(capture(outputs)))
            if affine is not None:
                handles.append(affine.register_forward_hook(capture(affine_outputs)))
            category = 'direction_audit_phase_loss_evaluations'
            self.budget[category] = self.budget.get(category, 0) + 1
            loss = phase_loss(trainer, role, *bank, step=anchor['step'])
            loss_value = float(loss.detach())
            report = {'player': role, 'step': anchor['step'], 'factor': float(factor),
                      'loss': loss_value if math.isfinite(loss_value) else None,
                      'epsilon': torch.finfo(loss.dtype).eps, 'gradient_dot_delta': None,
                      'gradient_dot_delta_absolute_sum': None,
                      'gradient_status': 'not_requested',
                      'derivative_definition': 'd/ds L(w+s*actual_optimizer_delta) evaluated at the requested factor on this bank'}
            if gradient:
                targets = [(parameter, delta) for parameter, delta in zip(owned, anchor['delta'])
                           if parameter.requires_grad]
                if loss.requires_grad and targets:
                    category = 'direction_audit_player_gradient_evaluations'
                    self.budget[category] = self.budget.get(category, 0) + 1
                    gradients = torch.autograd.grad(loss, [parameter for parameter, _ in targets], allow_unused=True)
                    slope, absolute_sum = 0., 0.
                    for value, (_, delta) in zip(gradients, targets):
                        if value is not None:
                            products = value.detach().cpu().double() * delta.double()
                            slope += float(products.sum())
                            absolute_sum += float(products.abs().sum())
                    connected = sum(value is not None for value in gradients)
                    report.update(gradient_dot_delta=slope if math.isfinite(slope) else None,
                                  gradient_dot_delta_absolute_sum=absolute_sum if math.isfinite(absolute_sum) else None,
                                  gradient_status=('disconnected' if not connected else
                                                   'finite' if math.isfinite(slope) else 'nonfinite'),
                                  gradient_parameter_tensors=connected,
                                  disconnected_parameter_tensors=len(targets) - connected)
                else:
                    report['gradient_status'] = 'disconnected'
            report['output'] = (_activation_scalars(outputs[0]) if len(outputs) == 1 and outputs[0] is not None else
                                {'status': 'skipped', 'reason': 'Generator did not produce exactly one tensor output'})
            if affine is not None:
                if len(affine_outputs) == 1 and affine_outputs[0] is not None:
                    description = {**description, 'activation': _activation_scalars(
                        affine_outputs[0], pre_tanh=description['relationship_to_output'] == 'pre_tanh')}
                else:
                    description = {**description, 'status': 'skipped',
                                   'reason': 'Final owned affine did not produce exactly one tensor output'}
            report['final_affine'] = description
            if _hash(parameters) != expected:
                raise ValueError('Direction audit changed registered parameters')
            if _hash(self.protected) != self.protected_hash:
                raise ValueError('Direction audit changed protected frozen/pretrained state')
            return report
        finally:
            for handle in handles:
                handle.remove()
            self.restore(trainer, entry)

    def gradient_change(self, anchor, role, bank, h, *, adam_denominators=None):
        """Two fixed-state player gradients and empirical directional bounds.

        The Cauchy quantities bound the observed secant's directional change,
        not an unobserved neighborhood's Lipschitz constant. This read-only
        research measurement does not propose a rate or alter tuning policy.
        """
        from .startup_dynamics import _snapshot
        if role not in ('generator', 'discriminator'):
            raise ValueError('Gradient-field audit requires generator or discriminator ownership')
        if type(h) not in (int, float) or not math.isfinite(h) or h <= 0:
            raise ValueError('Gradient-field audit h must be finite and positive')
        trainer = self.trainer
        owned = _parameters(trainer, role)
        denominators = None
        metric = {'status': 'not_supplied', 'reason': None}
        if adam_denominators is not None:
            if ('observation_step' in adam_denominators
                    and (type(adam_denominators['observation_step']) is not int
                         or adam_denominators['observation_step'] != anchor['step'])):
                raise ValueError('Adam metric observation step does not match the captured update anchor')
            metric = {'status': adam_denominators['status'], 'reason': adam_denominators.get('reason')}
            metric['provenance'] = ('captured_after_matching_observed_update' if 'observation_step' in adam_denominators
                                    else 'caller_provided; matching post-update capture is the caller responsibility')
            if metric['status'] == 'measured':
                names = {id(parameter): path for path, parameter in _registered_parameters(trainer)}
                denominators = adam_denominators['denominators']
                if (adam_denominators.get('player') != role
                        or adam_denominators.get('parameter_paths') != [names[id(parameter)] for parameter in owned]
                        or len(denominators) != len(owned)
                        or any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                               or value.shape != parameter.shape or not bool(torch.isfinite(value).all())
                               or not bool((value > 0).all()) for value, parameter in zip(denominators, owned))):
                    raise ValueError('Adam metric descriptor does not match the owned player parameters')
        entry = _snapshot(trainer)
        points = []
        realized_squared = 0.
        try:
            for factor in (0., h):
                self.restore(trainer, anchor['snapshot'])
                _copy(owned, [before + factor * delta for before, delta in zip(anchor['before'], anchor['delta'])])
                if factor == h:
                    realized_squared = sum(float((parameter.detach().cpu().double() - before.double()).square().sum())
                                           for parameter, before in zip(owned, anchor['before']))
                parameters = _registered_parameters(trainer)
                expected = _hash(parameters)
                category = 'gradient_field_phase_loss_evaluations'
                self.budget[category] = self.budget.get(category, 0) + 1
                loss = phase_loss(trainer, role, *bank, step=anchor['step'])
                targets = [(index, parameter) for index, parameter in enumerate(owned) if parameter.requires_grad]
                values = [None] * len(owned)
                if targets and loss.requires_grad:
                    category = 'gradient_field_player_gradient_evaluations'
                    self.budget[category] = self.budget.get(category, 0) + 1
                    gradients = torch.autograd.grad(loss, [parameter for _, parameter in targets], allow_unused=True)
                    for (index, _), gradient in zip(targets, gradients):
                        if gradient is not None:
                            values[index] = gradient.detach().cpu().clone()
                    del gradients
                    gradient = None
                if _hash(parameters) != expected:
                    raise ValueError('Gradient-field audit changed registered parameters')
                if _hash(self.protected) != self.protected_hash:
                    raise ValueError('Gradient-field audit changed protected frozen/pretrained state')
                points.append((float(loss.detach()), torch.finfo(loss.dtype).eps, values))
            squared_delta, squared_g0, squared_gh, squared_change = 0., 0., 0., 0.
            slope0, slope_h, absolute0, absolute_h = 0., 0., 0., 0.
            weighted_delta, weighted_change, weighted_g0, weighted_gh = 0., 0., 0., 0.
            for index, delta in enumerate(anchor['delta']):
                delta = delta.detach().cpu().double()
                g0, gh = points[0][2][index], points[1][2][index]
                g0 = torch.zeros_like(delta) if g0 is None else g0.double()
                gh = torch.zeros_like(delta) if gh is None else gh.double()
                difference = gh - g0
                squared_delta += float(delta.square().sum())
                squared_g0 += float(g0.square().sum())
                squared_gh += float(gh.square().sum())
                squared_change += float(difference.square().sum())
                product0, product_h = g0 * delta, gh * delta
                slope0 += float(product0.sum())
                slope_h += float(product_h.sum())
                absolute0 += float(product0.abs().sum())
                absolute_h += float(product_h.abs().sum())
                if denominators is not None:
                    denominator = denominators[index].double()
                    weighted_delta += float((delta.square() * denominator).sum())
                    weighted_change += float((difference.square() / denominator).sum())
                    weighted_g0 += float((g0.square() / denominator).sum())
                    weighted_gh += float((gh.square() / denominator).sum())
            def finite(value):
                return value if math.isfinite(value) else None
            report = {'status': 'finite', 'player': role, 'step': anchor['step'], 'h': float(h),
                      'loss0': finite(points[0][0]), 'loss_h': finite(points[1][0]),
                      'epsilon': max(point[1] for point in points),
                      'slope0': finite(slope0), 'slope_h': finite(slope_h),
                      'slope0_absolute_product_sum': finite(absolute0),
                      'slope_h_absolute_product_sum': finite(absolute_h),
                      'gradient0_norm': finite(math.sqrt(squared_g0)),
                      'gradient_h_norm': finite(math.sqrt(squared_gh)),
                      'gradient_change_norm': finite(math.sqrt(squared_change)),
                      'delta_norm': finite(math.sqrt(squared_delta)),
                      'realized_parameter_perturbation_norm': finite(math.sqrt(realized_squared)),
                      'directional_secant_curvature': finite((slope_h - slope0) / h),
                      'unweighted_cauchy_curvature': finite(math.sqrt(squared_delta) * math.sqrt(squared_change) / h),
                      'gradient0_connected_tensors': sum(value is not None for value in points[0][2]),
                      'gradient_h_connected_tensors': sum(value is not None for value in points[1][2]),
                      'interpretation': 'Measured two-point gradient-field response, not a certified neighborhood Lipschitz bound; no tuning decision is applied'}
            if any(value is None for key, value in report.items() if key not in ('adam_metric',)):
                report['status'] = 'nonfinite'
            elif realized_squared == 0.:
                report.update(status='unresolved', reason='Parameter perturbation rounded to zero')
            elif report['gradient0_connected_tensors'] == report['gradient_h_connected_tensors'] == 0:
                report.update(status='unresolved', reason='Player objective is disconnected from all owned parameters')
            if denominators is not None:
                metric.update(delta_metric_norm=finite(math.sqrt(weighted_delta)),
                              gradient_change_dual_norm=finite(math.sqrt(weighted_change)),
                              gradient0_dual_norm=finite(math.sqrt(weighted_g0)),
                              gradient_h_dual_norm=finite(math.sqrt(weighted_gh)),
                              cauchy_curvature=finite(math.sqrt(weighted_delta) * math.sqrt(weighted_change) / h),
                              definition='sqrt(sum(delta^2 * denominator)) * sqrt(sum((gh-g0)^2 / denominator)) / h',
                              denominator_definition='sqrt(bias_corrected_exp_avg_sq)+optimizer_eps')
                if any(metric[key] is None for key in ('delta_metric_norm', 'gradient_change_dual_norm', 'gradient0_dual_norm', 'gradient_h_dual_norm', 'cauchy_curvature')):
                    metric['status'] = 'nonfinite'
            report['adam_metric'] = metric
            return report
        finally:
            points.clear()
            self.restore(trainer, entry)

    def d_signal_response(self, anchor, banks):
        trainer = self.trainer
        terms = trainer.program.adversarial_terms
        supported = not trainer.program.objectives and all(
            all(route.path == 'candidate' for route in term.routes)
            and term.generator_phase.fake.path == 'generated'
            and not term.generator_phase.fake.detach_sample
            and not term.generator_phase.fake.detach_score
            and term.generator_phase.real.path == 'batch.real'
            for term in terms)
        if not supported:
            return {'status': 'skipped', 'reason': 'Fixed-image q requires attached generated fake samples, real data, candidate-only critic routes and no auxiliary objectives'}
        rows = []
        for bank in banks:
            gradients = []
            self.restore(trainer, anchor['snapshot'])
            try:
                expected = _hash(_registered_parameters(trainer))
                with torch.no_grad():
                    _, _, base_context = trainer._draw(*bank)
                self.budget['q_image_forwards'] += 1
                fixed_image = base_context['generated'].detach().clone()
                score_fence = _ephemeral(trainer)
                if _hash(_registered_parameters(trainer)) != expected:
                    raise ValueError('Fixed-image generation changed registered parameters')
                if _hash(self.protected) != self.protected_hash:
                    raise ValueError('Fixed-image generation changed protected frozen/pretrained state')
                for factor in (0., 1.):
                    self.restore(trainer, anchor['snapshot'])
                    _restore_ephemeral(trainer, score_fence)
                    _copy(_parameters(trainer, 'discriminator'), [before + factor * delta
                          for before, delta in zip(anchor['before'], anchor['delta'])])
                    expected = _hash(_registered_parameters(trainer))
                    image = fixed_image.clone().requires_grad_(True)
                    context = dict(base_context)
                    context['components'] = dict(base_context['components'])
                    context['prior'] = dict(base_context['prior'])
                    context['generated'] = image
                    context['components']['generator'] = image
                    weighted = []
                    for term in terms:
                        _, _, real_score, fake_score = _bound_scores(
                            term, context, trainer.graph, 'generator', term.generator_phase, first='fake')
                        weighted.append(term.weight * term.gan.g_loss(fake_score, real_score))
                    loss = _sum_tensors(weighted)
                    gradient = torch.autograd.grad(loss, image, allow_unused=True)[0] if loss.requires_grad else None
                    self.budget['d_signal_input_backwards'] += int(loss.requires_grad)
                    if _hash(_registered_parameters(trainer)) != expected:
                        raise ValueError('Fixed-image signal measurement changed registered parameters')
                    if _hash(self.protected) != self.protected_hash:
                        raise ValueError('Fixed-image signal measurement changed protected frozen/pretrained state')
                    if gradient is None:
                        return {'status': 'skipped', 'reason': 'Configured image objective is disconnected from the fixed image'}
                    gradients.append(gradient.detach().cpu())
            finally:
                self.restore(trainer, anchor['snapshot'])
            rows.append(tensor_change(*gradients))
        return {'status': 'measured', 'objective': 'configured_adversarial', 'step': anchor['step'],
                'input_control': 'identical_detached_image_real_context_and_scoring_randomness_before_after_D', 'banks': rows}
