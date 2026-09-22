"""Disposable, initialization-only generator gradient audit. No optimizer steps.

This numerical module is imported only by the explicit diagnose-signal command.
It uses the configured constructor/RNG order and native objective bindings, but
does not update D first: the reported phase is explicitly initial-G/frozen-D.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .config import config_values, fingerprint, load_config, resolve_config
from .objective_program import _bound_scores, _generator_tail, _sum_tensors
from .training import ReferenceTrainer


def _stats(value):
    """Retain only scalar reductions, never an activation/autograd graph."""
    value = value.detach()
    if not value.numel():
        return None
    value = value.to(dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)
    finite = torch.isfinite(value)
    safe = torch.where(finite, value, 0)
    return torch.stack((safe.square().mean().sqrt(), safe.mean(), safe.std(unbiased=False),
                        safe.abs().max(), (safe == 0).float().mean(),
                        (~finite).float().mean()))


def _numbers(stats):
    if stats is None:
        return None
    values = stats.cpu().tolist()
    keys = ('rms', 'mean', 'std', 'abs_max', 'zero_fraction', 'nonfinite_fraction')
    result = {key: value if math.isfinite(value) else None for key, value in zip(keys, values)}
    if result['nonfinite_fraction']:
        for key in keys[:-1]:
            result[key] = None
    return result


def _ratio(a, b):
    if a is None or b is None or b <= 0:
        return None
    value = a / b
    return value if math.isfinite(value) else None


def _tensors(value, suffix=''):
    if isinstance(value, torch.Tensor):
        yield suffix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensors(item, suffix + f'[{index}]')
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensors(item, suffix + f'[{key}]')


def _digest_state(modules):
    """Hash registered tensors, including nonpersistent buffers, one at a time."""
    digest = hashlib.sha256()
    for prefix, module in modules:
        for kind, values in (('parameter', module.named_parameters()), ('buffer', module.named_buffers())):
            for name, value in values:
                descriptor = (prefix, kind, name, tuple(value.shape), str(value.dtype))
                digest.update(json.dumps(descriptor).encode())
                raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                digest.update(raw.tobytes())
    return digest.hexdigest()


class _ActivationAudit:
    def __init__(self, generator, max_records=2048):
        self.rows, self.handles, self.calls = [], [], {}
        self.max_records = max_records
        for name, module in generator.named_modules():
            # HNDL operator nodes include semantic residual/attention boundaries.
            # Ordinary PyTorch models get leaf modules plus the generator output.
            node = '.nodes.' in '.' + name and name.rsplit('.', 1)[-1].startswith('n_')
            if not name or node or not tuple(module.children()):
                label = 'generator' + ('.' + name if name else '')
                self.handles.append(module.register_forward_hook(self._hook(label)))

    def _hook(self, name):
        def observe(module, args, output):
            invocation = self.calls.get(name, 0)
            self.calls[name] = invocation + 1
            for suffix, value in _tensors(output):
                if not value.is_floating_point():
                    continue
                if len(self.rows) >= self.max_records:
                    raise ValueError(f'Generator signal report exceeds {self.max_records} activation records')
                row = {'path': name + suffix, 'invocation': invocation,
                       'module_type': type(module).__name__, 'shape': list(value.shape),
                       'requires_grad': value.requires_grad, 'activation': _stats(value),
                       'gradient': None, 'backward_calls': 0}
                self.rows.append(row)
                if value.requires_grad:
                    def backward(gradient, row=row):
                        row['backward_calls'] += 1
                        row['gradient'] = _stats(gradient)
                    self.handles.append(value.register_hook(backward))
        return observe

    def close(self):
        for handle in self.handles:
            handle.remove()

    def finish(self, output_rms, batch_size):
        rows = []
        for source in self.rows:
            row = {**source, 'activation': _numbers(source['activation']),
                   'gradient': _numbers(source['gradient'])}
            gradient = row['gradient']
            if row['backward_calls'] > 1:
                raise ValueError('A repeated backward hook is unsupported by the initialization signal probe')
            row['status'] = ('no_autograd' if not row['requires_grad'] else
                             'not_used_by_objective' if gradient is None else
                             'nonfinite' if gradient['rms'] is None else
                             'zero_gradient' if gradient['rms'] == 0 else 'measured')
            magnitude = gradient['rms'] if gradient else None
            row['gradient_rms_times_batch_size'] = magnitude * batch_size if magnitude is not None else None
            row['gradient_to_output_rms_ratio'] = _ratio(magnitude, output_rms)
            activation = row['activation']
            row['activation_rms_times_gradient_rms'] = (
                magnitude * activation['rms']
                if magnitude is not None and activation and activation['rms'] is not None else None)
            rows.append(row)
        return rows


def _probe(trainer, objective, *, batch=None, latent_draw=None):
    """Audit an exclusively owned disposable trainer; restore flags and buffers."""
    graph, prior, program = trainer.graph, trainer.prior, trainer.program
    modules = (('graph', graph), ('prior', prior))
    before = _digest_state(modules)
    buffers = [(value, value.detach().clone()) for _, module in modules for value in module.buffers()]
    flags = [(p, p.requires_grad) for _, module in modules for p in module.parameters()]
    modes = {name: module.training for name, module in graph.named_modules()}
    generator_ids = {id(p) for p in program.generator_parameters}
    prior_ids = {id(p) for p in program.prior_parameters}
    inventory = [(name, p, 'generator_or_auxiliary' if id(p) in generator_ids else 'critic_or_frozen')
                 for name, p in graph.named_parameters()]
    inventory += [('prior.' + name, p, 'prior' if id(p) in prior_ids else 'frozen')
                  for name, p in prior.named_parameters()]
    original_flags = {id(p): flag for p, flag in flags}
    protected_buffers = set()
    for _, owner in modules:
        for module in owner.modules():
            parameters = tuple(module.parameters())
            if parameters and all(not original_flags[id(p)] for p in parameters):
                protected_buffers.update(id(value) for value in module.buffers())
    protected_buffers_changed = False
    audit = _ActivationAudit(graph.models['generator'])
    try:
        for term in program.adversarial_terms:
            term.module.requires_grad_(False)
        batch, ids, context = trainer._draw(batch, latent_draw)
        weighted = []
        terms = []
        for term in program.adversarial_terms:
            _, fake, real_score, fake_score = _bound_scores(
                term, context, graph, 'generator', term.generator_phase, first='fake')
            value = term.weight * term.gan.g_loss(fake_score, real_score)
            weighted.append(value)
            terms.append({'id': term.id, 'weight': term.weight, 'value': float(value.detach())})
        loss = _sum_tensors(weighted)
        if objective == 'total':
            prior_loss, auxiliary = _generator_tail(trainer, program, context, ids, fake)
            loss = loss + prior_loss + sum(auxiliary)
        generated = context['generated']
        if not loss.requires_grad:
            raise ValueError('The requested generator objective has no autograd path')
        output_target = [generated] if generated.requires_grad else []
        parameters = [p for _, p, _ in inventory if p.requires_grad]
        # Include the generated output to observe its gradient even when an
        # upstream route is detached. autograd.grad leaves parameter .grad alone.
        gradients = torch.autograd.grad(loss, output_target + parameters, allow_unused=True)
        output_gradient = gradients[0] if output_target else None
        output_stats = _numbers(_stats(output_gradient)) if output_gradient is not None else None
        by_id = {id(p): g for p, g in zip(parameters, gradients[len(output_target):])}
        parameter_rows = []
        for name, parameter, owner in inventory:
            gradient = by_id.get(id(parameter))
            if owner == 'critic_or_frozen':
                continue
            stats = _numbers(_stats(gradient)) if gradient is not None else None
            parameter_rows.append({'path': name, 'owner': owner, 'shape': list(parameter.shape),
                'parameter': _numbers(_stats(parameter)), 'gradient': stats,
                'status': ('frozen' if not original_flags[id(parameter)] else
                           'not_used_by_objective' if gradient is None else
                           'nonfinite' if stats['rms'] is None else
                           'zero_gradient' if stats['rms'] == 0 else 'measured')})
        output_rms = output_stats['rms'] if output_stats else None
        rows = audit.finish(output_rms, len(batch['real']))
        primary = [row for row in rows if row['invocation'] == 0 and row['gradient'] is not None]
        summary = {'generated_output_gradient_rms': output_rms,
                   'generated_output_gradient_rms_times_batch_size':
                       output_rms * len(batch['real']) if output_rms is not None else None,
                   'first_observed_module': primary[0]['path'] if primary else None,
                   'first_observed_to_output_rms_ratio': primary[0]['gradient_to_output_rms_ratio'] if primary else None,
                   'zero_gradient_activation_records': sum(r['status'] == 'zero_gradient' for r in rows),
                   'unused_activation_records': sum(r['status'] == 'not_used_by_objective' for r in rows),
                   'nonfinite_activation_records': sum(r['status'] == 'nonfinite' for r in rows)}
        result = {'loss': float(loss.detach()), 'adversarial_terms': terms, 'summary': summary,
                  'generated_output': {'shape': list(generated.shape), 'activation': _numbers(_stats(generated)),
                                       'gradient': output_stats},
                  'activations': rows, 'parameters': parameter_rows,
                  'frozen_parameter_elements': sum(p.numel() for p, flag in flags if not flag),
                  'module_training_modes': modes}
    finally:
        audit.close()
        with torch.no_grad():
            for value, saved in buffers:
                if id(value) in protected_buffers and not torch.equal(value, saved):
                    protected_buffers_changed = True
                value.copy_(saved)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
    if protected_buffers_changed:
        raise ValueError('Frozen module buffers changed during the signal probe; report refused')
    after = _digest_state(modules)
    if after != before:
        raise ValueError('Signal probe changed registered model state; report refused')
    result['state_verification'] = {'before_sha256': before, 'after_sha256': after,
                                   'parameters_and_restored_buffers_unchanged': True,
                                   'frozen_module_buffers_unchanged_during_probe': True,
                                   'optimizer_steps': 0, 'calibration_performed': False}
    return result


def diagnose(config_path, *, objective='adversarial', batch_size=None, device=None):
    """Construct fresh configured models and report one initial G backward pass.

    Process-local numerical policy and construction RNG are the trainer's own;
    call from a disposable CLI process, not from an active training process.
    """
    if objective not in ('adversarial', 'total'):
        raise ValueError('Signal objective must be adversarial or total')
    started = time.monotonic()
    config = load_config(config_path)
    original_fingerprint = fingerprint(config)
    original_batch = config['training']['batch_size']
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    if device is not None:
        config['training']['device'] = device
    config = resolve_config(config_values(config))
    trainer = ReferenceTrainer(config)
    # The probe uses online initialization; no EMA copies are needed afterwards.
    del trainer.ema_graph, trainer.ema_prior
    if trainer.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(trainer.device)
    result = _probe(trainer, objective)
    result.update(schema_version=1, kind='generator-initial-signal',
                  phase='initial-generator-frozen-discriminator-before-any-optimizer-step',
                  config_path=str(Path(config_path).resolve()), config_fingerprint=original_fingerprint,
                  effective_config_fingerprint=fingerprint(config), objective=objective,
                  seed=config['training']['seed'], configured_batch_size=original_batch,
                  probe_batch_size=config['training']['batch_size'], device=str(trainer.device),
                  torch_version=str(torch.__version__), seconds=time.monotonic() - started,
                  cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(trainer.device)
                                             if trainer.device.type == 'cuda' else None),
                  interpretation=['Initial frozen-critic derivative, not the first post-D-update training signal.',
                                  'Layer RMS ratios depend on activation coordinates and dimensions.',
                                  'Batch multiplication is a scale convention, not per-example gradients for coupled losses.',
                                  'This measures strength and transmission, not independent quality or optimal initialization.'])
    # Never publish nonfinite JSON, including a nonfinite objective.
    json.dumps(result, allow_nan=False)
    return result
