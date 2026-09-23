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


def _digest_state(modules, *, entries=None):
    """Hash registered tensors, optionally recording diagnostic hashes by path.

    Both digests share one CPU copy per tensor. The aggregate byte sequence is
    unchanged; entry hashes explain failures and never replace that guard.
    """
    digest = hashlib.sha256()
    for prefix, module in modules:
        for kind, values in (('parameter', module.named_parameters()), ('buffer', module.named_buffers())):
            for name, value in values:
                descriptor = (prefix, kind, name, tuple(value.shape), str(value.dtype))
                description = json.dumps(descriptor).encode()
                raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                digest.update(description)
                digest.update(raw)
                if entries is not None:
                    entry = hashlib.sha256(description)
                    entry.update(raw)
                    entries[f'{prefix}.{kind}.{name}'] = entry.hexdigest()
    return digest.hexdigest()


class _ActivationAudit:
    def __init__(self, generator, max_records=2048, *, label='generator', sample_order=None,
                 budget=None):
        from hndl.operators.pretrained import Pretrained
        self.rows, self.handles, self.calls = [], [], {}
        self.max_records = max_records
        self.budget = budget if budget is not None else [0]
        self.enabled = True
        self.sample_order, self.sample, self.forward_invocation = sample_order, None, -1
        if sample_order is not None:
            self.handles.append(generator.register_forward_pre_hook(self._begin, with_kwargs=True))
        pretrained = [name for name, module in generator.named_modules() if isinstance(module, Pretrained)]
        for name, module in generator.named_modules():
            if any(name != parent and (not parent or name.startswith(parent + '.')) for parent in pretrained):
                continue
            # HNDL operator nodes include semantic residual/attention boundaries.
            # Pretrained networks expose their interface, not thousands of internals.
            node = '.nodes.' in '.' + name and name.rsplit('.', 1)[-1].startswith('n_')
            path = label + ('.' + name if name else '')
            if sample_order is not None and (not name or name in pretrained):
                self.handles.append(module.register_forward_pre_hook(self._inputs(path), with_kwargs=True))
            if not name or node or name in pretrained or not tuple(module.children()):
                self.handles.append(module.register_forward_hook(self._hook(path)))

    def _begin(self, module, args, kwargs):
        if self.enabled:
            self.forward_invocation += 1
            self.sample = (self.sample_order[self.forward_invocation]
                           if self.forward_invocation < len(self.sample_order) else 'additional')

    def _record(self, name, module, value, invocation, boundary):
        for suffix, tensor in _tensors(value):
            if not tensor.is_floating_point():
                continue
            if self.budget[0] >= self.max_records:
                raise ValueError(f'Signal report exceeds {self.max_records} activation records')
            self.budget[0] += 1
            row = {'path': name + suffix, 'invocation': invocation,
                   'module_type': type(module).__name__, 'shape': list(tensor.shape),
                   'requires_grad': tensor.requires_grad, 'activation': _stats(tensor),
                   'gradient': None, 'backward_calls': 0, 'boundary': boundary}
            if self.sample_order is not None:
                row.update(sample=self.sample, critic_invocation=self.forward_invocation)
            self.rows.append(row)
            if tensor.requires_grad:
                def backward(gradient, row=row):
                    row['backward_calls'] += 1
                    row['gradient'] = _stats(gradient)
                self.handles.append(tensor.register_hook(backward))

    def _inputs(self, name):
        def observe(module, args, kwargs):
            if self.enabled:
                self._record(name + '.input', module, {'args': args, 'kwargs': kwargs},
                             self.forward_invocation, 'input')
        return observe

    def _hook(self, name):
        def observe(module, args, output):
            if not self.enabled:
                return
            invocation = self.calls.get(name, 0)
            self.calls[name] = invocation + 1
            self._record(name, module, output, invocation, 'output')
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
    before_entries = {}
    before = _digest_state(modules, entries=before_entries)
    buffers = [(f'{prefix}.buffer.{name}', value, value.detach().clone())
               for prefix, module in modules for name, value in module.named_buffers()]
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
    protected_buffers_changed = []
    audit = _ActivationAudit(graph.models['generator'])
    critic_audits, critic_budget = [], [0]
    try:
        for term in program.adversarial_terms:
            term.module.requires_grad_(False)
        batch, ids, context = trainer._draw(batch, latent_draw)
        weighted = []
        terms = []
        for term in program.adversarial_terms:
            label = 'discriminator[' + term.id + ']'
            critic_audit = _ActivationAudit(term.module, label=label, sample_order=('fake', 'real'),
                                            budget=critic_budget)
            critic_audits.append((term, label, critic_audit))
            _, fake, real_score, fake_score = _bound_scores(
                term, context, graph, 'generator', term.generator_phase, first='fake')
            critic_audit.enabled = False
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
        critic_rows, critic_profiles = [], []
        for term, label, critic_audit in critic_audits:
            measured = critic_audit.finish(output_rms, len(batch['real']))
            scores = [row for row in measured if row['path'] == label and row['sample'] == 'fake']
            score = scores[0] if scores else None
            score_rms = score['gradient']['rms'] if score and score['gradient'] else None
            for row in measured:
                row['term_id'] = term.id
                row['gradient_to_generated_output_rms_ratio'] = row.pop('gradient_to_output_rms_ratio')
                row['gradient_to_fake_score_rms_ratio'] = (
                    _ratio(row['gradient']['rms'], score_rms)
                    if row['sample'] == 'fake' and row['gradient'] else None)
            critic_rows.extend(measured)
            inputs = [row for row in measured if row['boundary'] == 'input'
                      and row['path'].startswith(label + '.input[') and row['sample'] == 'fake']
            critic_profiles.append({
                'term_id': term.id, 'weight': term.weight,
                'fake_binding': {'path': term.generator_phase.fake.path,
                                 'detach_sample': term.generator_phase.fake.detach_sample,
                                 'detach_score': term.generator_phase.fake.detach_score},
                'real_binding': {'path': term.generator_phase.real.path,
                                 'detach_sample': term.generator_phase.real.detach_sample,
                                 'detach_score': term.generator_phase.real.detach_score},
                'fake_score_activation': score['activation'] if score else None,
                'fake_score_gradient_rms': score_rms,
                'fake_score_status': score['status'] if score else 'unavailable',
                'fake_input_gradients': [{'path': row['path'], 'status': row['status'],
                    'gradient_rms': row['gradient']['rms'] if row['gradient'] else None,
                    'gradient_to_score_rms_ratio': _ratio(row['gradient']['rms'], score_rms)
                        if row['gradient'] else None} for row in inputs],
                'nonfinite_activation_records': sum(row['status'] == 'nonfinite' for row in measured),
                'zero_gradient_activation_records': sum(row['status'] == 'zero_gradient' for row in measured),
            })
        summary['discriminator_profiles'] = critic_profiles
        result = {'loss': float(loss.detach()), 'adversarial_terms': terms, 'summary': summary,
                  'generated_output': {'shape': list(generated.shape), 'activation': _numbers(_stats(generated)),
                                       'gradient': output_stats},
                  'activations': rows, 'parameters': parameter_rows,
                  'discriminator_activations': critic_rows,
                  'discriminator_interpretation': [
                      'Measured during the same generator-objective backward with critic parameters frozen.',
                      'Fake and real identify score invocation, not parameter updates; detached real branches can have no gradient.',
                      'Shared input tensors carry the summed objective gradient, not an isolated per-critic derivative.',
                      'Magnitude ratios depend on dimensions, coordinates, term weights and loss; no universal optimal norm is implied.',
                      'Pretrained wrappers expose input/output boundaries; their internal layers are not instrumented.'],
                  'frozen_parameter_elements': sum(p.numel() for p, flag in flags if not flag),
                  'module_training_modes': modes}
    finally:
        audit.close()
        for _, _, critic_audit in critic_audits:
            critic_audit.close()
        with torch.no_grad():
            for path, value, saved in buffers:
                if id(value) in protected_buffers and not torch.equal(value, saved):
                    protected_buffers_changed.append(path)
                value.copy_(saved)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
    if protected_buffers_changed:
        raise ValueError('Frozen module buffers changed during the signal probe; report refused; '
                         'changed registered tensors: ' + ', '.join(protected_buffers_changed))
    after_entries = {}
    after = _digest_state(modules, entries=after_entries)
    if after != before:
        changed = sorted(path for path in before_entries.keys() | after_entries.keys()
                         if before_entries.get(path) != after_entries.get(path))
        paths = ', '.join(changed) or '(none identified; aggregate mismatch remains fatal)'
        raise ValueError('Signal probe changed registered model state; report refused; '
                         f'changed registered tensors: {paths}; '
                         f'before_sha256={before}; after_sha256={after}')
    result['state_verification'] = {'before_sha256': before, 'after_sha256': after,
                                   'parameters_and_restored_buffers_unchanged': True,
                                   'frozen_module_buffers_unchanged_during_probe': True,
                                   'optimizer_steps': 0, 'calibration_performed': False}
    return result


def diagnose(config_path, *, objective='adversarial', batch_size=None, device=None, checkpoint=None):
    """Measure fresh initialization or an isolated full online G/D checkpoint.

    A run source is read only: pin and validate a completed checkpoint, restore
    into disposable models, and probe without a controller, optimizer update,
    checkpoint save, or live-training mutation. Call from a disposable process.
    """
    if objective not in ('adversarial', 'total'):
        raise ValueError('Signal objective must be adversarial or total')
    started = time.monotonic()
    source = Path(config_path).resolve()
    if source.suffix in ('.pt', '.pth', '.safetensors'):
        raise ValueError('Signal diagnosis needs the full online generator and discriminator; '
                         'pass a run directory with a complete training checkpoint, not an inference model')
    saved = None
    target = None
    is_run = source.is_dir() and (source / 'manifest.json').is_file()
    if checkpoint is not None and not is_run:
        raise ValueError('--checkpoint requires a run directory containing complete training checkpoints')
    if is_run:
        from .signal_checkpoint import load_signal_checkpoint
        trainer, original, target, saved, protocol = load_signal_checkpoint(
            source, checkpoint, device=device, batch_size=batch_size)
        config = trainer.config
        original_fingerprint = fingerprint(original)
        original_batch = original['training']['batch_size']
    else:
        config = load_config(source)
        original_fingerprint = fingerprint(config)
        original_batch = config['training']['batch_size']
        if batch_size is not None:
            config['training']['batch_size'] = batch_size
        if device is not None:
            config['training']['device'] = device
        config = resolve_config(config_values(config))
        trainer = ReferenceTrainer(config)
    # Online generator and discriminator are the source; EMA is not probed.
    del trainer.ema_graph, trainer.ema_prior
    if trainer.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(trainer.device)
    if trainer.device.type == 'cuda':
        with torch.cuda.device(trainer.device):
            result = _probe(trainer, objective)
    else:
        result = _probe(trainer, objective)
    initial = saved is None
    phase = ('initial-generator-frozen-discriminator-before-any-optimizer-step' if initial else
             f'frozen-checkpoint-generator-and-discriminator-at-step-{trainer.step}')
    result.update(schema_version=1,
                  kind='generator-initial-signal' if initial else 'generator-checkpoint-signal',
                  phase=phase, step=trainer.step, source_path=str(source),
                  config_fingerprint=original_fingerprint,
                  effective_config_fingerprint=fingerprint(config), objective=objective,
                  seed=config['training']['seed'], configured_batch_size=original_batch,
                  probe_batch_size=config['training']['batch_size'], device=str(trainer.device),
                  torch_version=str(torch.__version__), seconds=time.monotonic() - started,
                  cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(trainer.device)
                                             if trainer.device.type == 'cuda' else None),
                  interpretation=[('Initial frozen-critic derivative, not the first post-D-update training signal.'
                                   if initial else 'Frozen online G/D checkpoint derivative; no optimizer update, EMA substitution or live training mutation.'),
                                  'Layer RMS ratios depend on activation coordinates and dimensions.',
                                  'Batch multiplication is a scale convention, not per-example gradients for coupled losses.',
                                  'This measures strength and transmission, not independent quality or optimal initialization.'])
    if initial:
        result['config_path'] = str(source)
    else:
        result['checkpoint'] = {
            'path': str(target), 'step': trainer.step, 'run_id': saved['run_id'],
            'attempt_id': saved['attempt_id'], 'state_sha256': saved['state_sha256'],
            'config_sha256': saved['config_sha256'], 'source': saved.get('source'),
            'manifest_sha256': hashlib.sha256((target / 'manifest.json').read_bytes()).hexdigest(),
            'weights': 'online-generator-and-discriminator',
            'initialization_tuning': saved.get('initialization_tuning'),
            'diagnostic_protocol': protocol,
        }
    json.dumps(result, allow_nan=False)
    return result
