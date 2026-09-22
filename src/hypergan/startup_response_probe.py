"""Private phase-local measurements for the disposable startup update trial."""
import math

import torch

from .checkpoints import capture_rng, restore_rng
from .objective_program import _bound_scores, _generator_tail, _sum_tensors, score_candidate
from .signal_structure import _hash
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


class UpdateObserver:
    """Observe only updates one/eight; save phase-eight anchors, never graphs."""
    def __init__(self, trainer, snapshot, protected, protected_hash, budget, *, capture_anchors=True):
        self.trainer, self.snapshot = trainer, snapshot
        self.protected, self.protected_hash, self.budget = protected, protected_hash, budget
        self.anchors, self.observations, self.pending = {}, [], {}
        self.capture_anchors = capture_anchors

    def __call__(self, event, *, step, batch, ids, context):
        if step not in (1, 8):
            return
        role = 'discriminator' if event.endswith('_d') else 'generator'
        parameters = _parameters(self.trainer, role)
        if event.startswith('before'):
            entry = {'before': _values(parameters)}
            if role == 'generator':
                entry['prior_before'] = _values(_parameters(self.trainer, 'prior'))
                entry['latent'] = context['latent'].detach().clone()
            if step == 8 and self.capture_anchors:
                entry['snapshot'] = self.snapshot(self.trainer)
            self.pending[role] = entry
            return
        entry = self.pending.pop(role)
        deltas, motion = displacement(parameters, entry['before'])
        row = {'step': step, 'player': role, 'optimizer_motion': motion}
        if step == 8 and self.capture_anchors:
            self.anchors[role] = {'snapshot': entry['snapshot'], 'before': entry['before'],
                                  'delta': deltas, 'step': step}
        if role == 'generator':
            _, row['prior_optimizer_motion'] = displacement(_parameters(self.trainer, 'prior'), entry['prior_before'])
            # Two replay forwards avoid assuming the original stochastic forward
            # can be reused. Fixed latent values deliberately exclude prior motion.
            after = _values(parameters)
            fence = _ephemeral(self.trainer)
            try:
                outputs = []
                for values in (entry['before'], after):
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
                row['generator_output_response'] = _image_response(*outputs)
                row['input_control'] = 'fixed_latent_values; prior output response not included'
            finally:
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
