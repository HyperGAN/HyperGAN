"""Bounded startup-only scale search, never a claim of GAN signal quality.

Only the first and final executed, owned affine layers are candidates. A fixed
output cotangent measures transmission independently of critic magnitude. The
configured GAN derivative is reported separately. No optimizer steps are taken.
"""
import copy
import hashlib
import json
import math

import torch
from torch import nn

from .checkpoints import capture_rng, restore_rng, data_contract
from .signal_diagnostic import _probe


def _storage(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


def _inventory(trainer):
    """Fail closed on external/unknown ownership, frozen tensors and aliases."""
    from hndl.operators.pretrained import Pretrained
    generator = trainer.graph.models['generator']
    if trainer.config['components']['generator']['factory'] != 'hndl':
        return [], 'Only native HNDL generator ownership is supported for calibration.'
    registered = []
    for prefix, root in (('graph', trainer.graph), ('prior', trainer.prior)):
        registered += [(prefix + '.' + n, p) for n, p in root.named_parameters(remove_duplicate=False)]
        registered += [(prefix + '.' + n, p) for n, p in root.named_buffers(remove_duplicate=False)]
    aliases = {}
    for name, tensor in registered:
        aliases.setdefault(_storage(tensor), []).append(name)
    forbidden = set()
    for root in (trainer.graph, trainer.prior):
        for module in root.modules():
            if isinstance(module, Pretrained):
                forbidden.update(_storage(p) for p in list(module.parameters()) + list(module.buffers()))
    owned = {id(p) for p in trainer.program.generator_parameters}
    layers = []
    for name, module in generator.named_modules():
        # Arbitrary subclasses can load foreign weights in their constructors.
        recognized = (type(module) in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
                      or type(module).__module__.startswith('hndl.operators.'))
        if not recognized or not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue
        ancestors = [generator.get_submodule('.'.join(name.split('.')[:i]))
                     for i in range(1, len(name.split('.')))]
        if any(not type(parent).__module__.startswith(('hndl.', 'torch.nn.'))
               for parent in ancestors):
            continue
        parameters = list(module.named_parameters(recurse=False))
        if not parameters or any(not p.requires_grad or id(p) not in owned or _storage(p) in forbidden
                                 or len(aliases[_storage(p)]) != 1 for _, p in parameters):
            continue
        layers.append((name, module))
    return layers, None


def _tensors(trainer):
    for prefix, root in (('graph', trainer.graph), ('prior', trainer.prior)):
        for kind, values in (('parameter', root.named_parameters()), ('buffer', root.named_buffers())):
            for name, value in values:
                yield prefix + '.' + kind + '.' + name, value


def _hash(tensors):
    digest = hashlib.sha256()
    for name, value in tensors:
        digest.update(json.dumps((name, list(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _structural(trainer, batch, latent, layers):
    activations, handles = [], []
    for name, module in layers:
        def observe(module, args, output, name=name):
            if isinstance(output, torch.Tensor) and output.requires_grad:
                activations.append((name, output))
        handles.append(module.register_forward_hook(observe))
    try:
        _, _, context = trainer._draw(batch, latent)
        output = context['generated']
        # One deterministic isotropic cotangent; no global RNG or seed sweep.
        rng = torch.Generator(device='cpu').manual_seed(0)
        cotangent = torch.randint(0, 2, output.shape, generator=rng, dtype=torch.int8).to(output.device, output.dtype) * 2 - 1
        targets = [value for _, value in activations]
        gradients = torch.autograd.grad(output, targets, grad_outputs=cotangent, allow_unused=True) if targets else []
        output_rms = float(output.detach().float().square().mean().sqrt())
        rows = []
        for (name, value), gradient in zip(activations, gradients):
            rms = float(value.detach().float().square().mean().sqrt())
            grad_rms = None if gradient is None else float(gradient.detach().float().square().mean().sqrt())
            # Relative-coordinate sensitivity. This removes trivial reciprocal
            # rescaling of an activation and its downstream weights, but is not
            # architecture-independent dynamical isometry.
            gain = (rms * grad_rms * math.sqrt(value.numel() / output.numel())
                    / max(output_rms, 1e-12)) if grad_rms is not None else None
            rows.append({'path': name, 'activation_rms': rms, 'gradient_rms': grad_rms,
                         'relative_cotangent_gain': gain})
        edges = [rows[0], rows[-1]] if len(rows) > 1 else rows
        valid = bool(edges) and all(r['relative_cotangent_gain'] is not None
                                   and math.isfinite(r['relative_cotangent_gain'])
                                   and r['relative_cotangent_gain'] > 0 for r in edges)
        score = sum(abs(math.log10(r['relative_cotangent_gain'])) for r in edges) / len(edges) if valid else None
        value = output.detach().float()
        result = {'score': score, 'affine_layers': rows, 'output_rms': output_rms,
                  'output_std': float(value.std(unbiased=False)),
                  'sample_diversity_rms': float(value.var(dim=0, unbiased=False).mean().sqrt()),
                  'nonfinite_output_fraction': float((~torch.isfinite(value)).float().mean()),
                  'absolute_output_above_0_99_fraction': float((value.abs() > .99).float().mean())}
        return result
    finally:
        for handle in handles:
            handle.remove()


def _acceptable(baseline, candidate, diagnostic):
    reasons = []
    if candidate['score'] is None or not math.isfinite(candidate['score']):
        reasons.append('invalid_or_disconnected_cotangent_signal')
    elif baseline['score'] is not None and candidate['score'] >= baseline['score'] - max(.01, baseline['score'] * .05):
        reasons.append('transmission_score_did_not_improve_by_five_percent')
    if candidate['nonfinite_output_fraction']:
        reasons.append('nonfinite_output')
    for key in ('output_rms', 'output_std', 'sample_diversity_rms'):
        old, new = baseline[key], candidate[key]
        if not math.isfinite(new) or new < max(1e-8, old * .5) or new > max(1e-8, old * 2):
            reasons.append(key + '_outside_guard')
    if not math.isfinite(diagnostic['loss']):
        reasons.append('nonfinite_objective')
    if diagnostic['summary']['nonfinite_activation_records']:
        reasons.append('nonfinite_objective_gradient')
    parameters = diagnostic.get('parameters', [])
    if any(row['status'] == 'nonfinite' for row in parameters):
        reasons.append('nonfinite_parameter_gradient')
    generator_rows = [row for row in parameters if row['owner'] == 'generator_or_auxiliary'
                      and row['path'].startswith('models.generator.') and row['status'] != 'frozen']
    if not any(row['status'] == 'measured' for row in generator_rows):
        reasons.append('missing_generator_parameter_signal')
    if any(row.get('activation', {}).get('nonfinite_fraction', 0)
           for row in diagnostic.get('activations', []) if row.get('activation')):
        reasons.append('nonfinite_activation')
    if diagnostic['summary']['generated_output_gradient_rms'] in (None, 0):
        reasons.append('missing_objective_signal')
    return reasons


def tune_initialization(trainer, *, objective='adversarial', progress=None, max_candidates=3):
    """Evaluate fixed draws and retain a safe bounded boundary-affine scale edit.

    All global/stream/data RNG, modes, buffers and gradient fields are preserved.
    Caller must synchronize initial EMA after success and persist the exact tuned
    tensors. This function is legal only before any training/optimizer update.
    """
    if trainer.step != 0 or trainer.opt_g.state or trainer.opt_d.state:
        raise ValueError('Initialization tuning is allowed only before the first optimizer update')
    if not 1 <= max_candidates <= 3:
        raise ValueError('Initialization tuning supports one to three candidates')
    contract = data_contract(trainer.data, trainer.config['data'])
    if not contract['supported']:
        raise ValueError('Initialization tuning requires restorable or declared stateless data')
    layers, exclusion = _inventory(trainer)
    rng = capture_rng()
    streams = {name: stream.get_state().clone() for name, stream in trainer.streams.items()}
    data = copy.deepcopy(trainer.data.state_dict()) if contract['stateful'] else None
    saved = [(name, value, value.detach().cpu().clone()) for name, value in _tensors(trainer)]
    modes = [(m, m.training) for root in (trainer.graph, trainer.prior) for m in root.modules()]
    buffers = [(v, v.detach().clone()) for root in (trainer.graph, trainer.prior) for v in root.buffers()]
    from hndl.operators.pretrained import Pretrained
    pretrained = [(name, tensor) for prefix, root in (('graph', trainer.graph), ('prior', trainer.prior))
                  for path, module in root.named_modules() if isinstance(module, Pretrained)
                  for name, tensor in [(prefix + '.' + path + '.' + n, p)
                                       for n, p in list(module.named_parameters()) + list(module.named_buffers())]]
    pretrained_before = _hash(pretrained)
    selected, completed = {}, False
    progress = progress or (lambda row: None)
    def restore_buffers():
        with torch.no_grad():
            for value, original in buffers:
                value.copy_(original)
    def evaluate(batch, latent):
        restore_rng(probe_rng)
        parameter_tensors = [(name, value) for name, value, _ in saved if '.parameter.' in name]
        parameter_before = _hash(parameter_tensors)
        structural = _structural(trainer, batch, latent, layers)
        if _hash(parameter_tensors) != parameter_before:
            raise ValueError('Initialization probe changed model parameters; restoring baseline')
        if _hash(pretrained) != pretrained_before:
            raise ValueError('Initialization probe changed pretrained state; restoring baseline')
        restore_buffers()
        restore_rng(probe_rng)
        diagnostic = _probe(trainer, objective, batch=batch, latent_draw=latent)
        restore_buffers()
        return {'transmission': structural, 'signal': diagnostic}
    try:
        progress({'candidate': 0, 'total_candidates': max_candidates, 'message': 'Measuring initial generator signal'})
        batch = trainer.batch()
        # Freeze prior draws without retaining a graph into its particle table.
        with torch.no_grad():
            latent = trainer.prior.sample(len(batch['real']), generator=trainer.streams['prior'])
        probe_rng = capture_rng()
        baseline = evaluate(batch, latent)
        observed = baseline['transmission']['affine_layers']
        # Only once-executed boundary layers are safe automatic scale targets.
        by_name = dict(layers)
        boundary = [observed[0], observed[-1]] if len(observed) > 1 else observed
        proposals = {}
        for row in boundary:
            name, rms = row['path'], row['activation_rms']
            if sum(r['path'] == name for r in observed) != 1 or not math.isfinite(rms) or rms <= 1e-12:
                continue
            target = .7 if row is boundary[-1] else 1.
            gain = min(2., max(.5, target / rms))
            for local_name, p in by_name[name].named_parameters(recurse=False):
                proposals[id(p)] = ('models.generator.' + name + '.' + local_name, p, gain)
        allowed_ids = set(proposals)
        protected = [(name, p) for name, p, _ in saved if id(p) not in allowed_ids]
        protected_before = _hash(protected)
        original = {id(p): value for _, p, value in saved}
        candidates = []
        best = baseline
        best_score = baseline['transmission']['score']
        selected_name = 'baseline'
        for index, design in enumerate(('output_boundary', 'input_boundary', 'both_boundaries')[:max_candidates], 1):
            if not proposals:
                break
            progress({'candidate': index, 'total_candidates': max_candidates,
                      'message': 'Evaluating bounded generator scale candidate ' + str(index)})
            candidate_factors = {}
            boundary_path = observed[-1]['path'] if design == 'output_boundary' else observed[0]['path']
            with torch.no_grad():
                for key, (path, p, gain) in proposals.items():
                    active = design == 'both_boundaries' or path.rsplit('.', 1)[0] == 'models.generator.' + boundary_path
                    factor = gain if active else 1.
                    p.copy_(original[id(p)]).mul_(factor)
                    if factor != 1.:
                        candidate_factors[key] = factor
            measurement = evaluate(batch, latent)
            reasons = _acceptable(baseline['transmission'], measurement['transmission'], measurement['signal'])
            score = measurement['transmission']['score']
            accepted = not reasons and (best_score is None or score < best_score)
            name = design
            candidates.append({'name': name, 'eligible': not reasons,
                               'rejection_reasons': reasons, **measurement})
            if accepted:
                selected = candidate_factors
                selected_name, best, best_score = name, measurement, score
            if _hash(protected) != protected_before:
                raise ValueError('Initialization tuning changed protected state; restoring baseline')
        # Confirm the selected edit on a second independent draw from the same
        # configured stream. This is held-out data, not a seed experiment.
        validation = None
        if selected:
            progress({'candidate': len(candidates), 'total_candidates': max_candidates,
                      'message': 'Confirming selected scale on a held-out startup batch'})
            holdout_batch = trainer.batch()
            with torch.no_grad():
                holdout_latent = trainer.prior.sample(len(holdout_batch['real']), generator=trainer.streams['prior'])
                for _, p, _ in proposals.values():
                    p.copy_(original[id(p)])
            holdout_before = evaluate(holdout_batch, holdout_latent)
            with torch.no_grad():
                for key, factor in selected.items():
                    proposals[key][1].mul_(factor)
            holdout_after = evaluate(holdout_batch, holdout_latent)
            reasons = _acceptable(holdout_before['transmission'], holdout_after['transmission'], holdout_after['signal'])
            validation = {'before': holdout_before, 'after': holdout_after,
                          'accepted': not reasons, 'rejection_reasons': reasons}
            if reasons:
                selected, selected_name, best = {}, 'baseline', baseline
        with torch.no_grad():
            for _, p, _ in proposals.values():
                p.copy_(original[id(p)])
                if id(p) in selected:
                    p.mul_(selected[id(p)])
        protected_after = _hash(protected)
        if protected_after != protected_before:
            raise ValueError('Initialization tuning changed protected state; restoring baseline')
        result = {'schema_version': 1, 'kind': 'generator-initialization-tuning',
                  'outcome': 'selected' if selected else 'kept_baseline', 'selected_candidate': selected_name,
                  'before': baseline, 'after': best, 'candidates': candidates, 'heldout_validation': validation,
                  'transformations': [{'path': proposals[key][0], 'factor': factor} for key, factor in selected.items()],
                  'protected_state_verification': {'before_sha256': protected_before, 'after_sha256': protected_after,
                                                   'unchanged': True},
                  'optimizer_steps': 0, 'ownership_exclusion': exclusion,
                  'interpretation': ['A bounded startup heuristic, not a universal optimum or proof of useful GAN learning.',
                                     'Fixed cotangent tests local transmission; configured objective gradients are reported independently.',
                                     'Only owned first/final affine weights and biases may change; pretrained tensors, buffers, prior, critic and normalization are protected.',
                                     'Scores depend on activation coordinates and architecture; selection uses one batch and confirmation a second held-out draw.',
                                     'Initialization improvements can drift during training.']}
        json.dumps(result, allow_nan=False)
        completed = True
        return result
    finally:
        with torch.no_grad():
            for _, value, original in saved:
                if not completed or id(value) not in selected:
                    value.copy_(original)
        for module, mode in modes:
            module.training = mode
        for name, state in streams.items():
            trainer.streams[name].set_state(state)
        if contract['stateful']:
            trainer.data.load_state_dict(data)
        restore_rng(rng)
