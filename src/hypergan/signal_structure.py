"""Read-only generator transmission and conservative ownership inventory."""
import hashlib
import json
import math

import torch
from torch import nn


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
