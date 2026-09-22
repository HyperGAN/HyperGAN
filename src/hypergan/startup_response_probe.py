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
                                  'delta': deltas, 'step': step}
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
