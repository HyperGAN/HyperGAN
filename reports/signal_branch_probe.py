"""Read-only, architecture-specific DINO multidepth discriminator attribution.

Research helper, not a public HyperGAN metric or calibration policy. Call only
with an exclusively owned disposable trainer and explicit matched probe inputs.
"""
import json
import math

import torch
import torch.nn.functional as F

from hypergan.checkpoints import capture_rng, restore_rng
from hypergan.objective_program import _bound_scores
from hypergan.signal_diagnostic import _digest_state


COEFFICIENTS = {'pixel_score': 1 / math.sqrt(2),
                **{f'feature{i}_score': .5 / math.sqrt(2) for i in range(1, 5)}}


def _rms(value):
    return float(value.detach().double().square().mean().sqrt())


def _norm(value):
    return float(value.detach().double().square().sum().sqrt())


def _cosine(left, right):
    scale = _norm(left) * _norm(right)
    return float((left.detach().double() * right.detach().double()).sum()) / scale if scale else None


def _score_stats(value):
    value = value.detach().double()
    return {'mean': float(value.mean()), 'std': float(value.std(unbiased=False)),
            'min': float(value.min()), 'max': float(value.max()), 'rms': _rms(value)}


def _image_stats(value):
    value = value.detach().float()
    rows = value.flatten(1)
    norms = rows.norm(dim=1)
    valid = norms > 0
    unit = rows[valid] / norms[valid, None]
    n = len(unit)
    pairwise = float((unit.sum(dim=0).double().square().sum() - n) / (n * (n - 1))) if n > 1 else None
    energy = float(value.double().square().sum())
    frequencies = {}
    for size in (2, 4, 8, 16):
        if any(dimension % size for dimension in value.shape[-2:]):
            continue
        pooled = F.avg_pool2d(value, size, stride=size)
        low = F.interpolate(pooled, size=value.shape[-2:], mode='nearest')
        frequencies[str(size)] = {
            'block_mean_gradient_rms': _rms(pooled),
            'low_frequency_energy_fraction': float(low.double().square().sum()) / energy if energy else None,
            'residual_high_frequency_energy_fraction': float((value - low).double().square().sum()) / energy if energy else None,
        }
    return {'rms': _rms(value), 'l2_norm': _norm(value),
            'per_sample_norm_mean': float(norms.mean()), 'per_sample_norm_std': float(norms.std(unbiased=False)),
            'sample_coordinate_pairwise_cosine_mean': pairwise,
            'sample_coordinate_alignment_valid_rows': n,
            'pooled_spatial_gradient': frequencies}


def _architecture(trainer):
    terms = trainer.program.adversarial_terms
    if len(terms) != 1:
        raise ValueError('Branch probe requires exactly one multidepth discriminator objective')
    term = terms[0]
    phase = term.generator_phase
    if (phase.fake.path != 'generated' or phase.real.path != 'batch.real'
            or phase.fake.detach_sample or phase.fake.detach_score
            or len(term.routes) != 1 or term.routes[0].path != 'candidate'):
        raise ValueError('Branch probe requires direct generated/real image bindings and one candidate-only discriminator route')
    modules = {}
    for branch in COEFFICIENTS:
        found = [(name, module) for name, module in term.module.named_modules()
                 if name.rsplit('.', 1)[-1] == 'n_' + branch]
        if len(found) != 1:
            raise ValueError(f'Expected one named multidepth branch n_{branch}; found {len(found)}')
        modules[branch] = found[0]
    return term, modules


def probe_branches(trainer, batch, latent_draw):
    """Decompose the actual adversarial image cotangent into five score branches.

    G is evaluated without a backward graph; its image becomes an independent
    leaf. Therefore the result attributes dL/dimage through D and says nothing
    about dL/dG parameters. The actual configured GAN objective supplies branch
    cotangents. No surrogate objective, optimizer update or pretrained edit occurs.
    """
    if batch is None or latent_draw is None:
        raise ValueError('Branch attribution requires explicit matched real and latent draws')
    term, modules = _architecture(trainer)
    roots = (('graph', trainer.graph), ('prior', trainer.prior))
    before = _digest_state(roots)
    rng = capture_rng()
    streams = {name: stream.get_state().clone() for name, stream in trainer.streams.items()}
    buffers = [(value, value.detach().clone()) for _, root in roots for value in root.buffers()]
    flags = [(parameter, parameter.requires_grad) for _, root in roots for parameter in root.parameters()]
    modes = [(module, module.training) for _, root in roots for module in root.modules()]
    captured = {name: [] for name in modules}
    handles = []
    result = None
    try:
        for parameter in term.module.parameters():
            parameter.requires_grad_(False)
        # No sampler is advanced: _draw receives both explicit arguments.
        with torch.no_grad():
            _, _, context = trainer._draw(batch, latent_draw)
        image = context['generated'].detach().requires_grad_(True)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Multidepth image attribution requires batched RGB images')
        context['generated'] = image
        for name, (_, module) in modules.items():
            def capture(module, args, output, name=name):
                if not isinstance(output, torch.Tensor):
                    raise ValueError('Multidepth score head must return a tensor')
                captured[name].append(output)
            handles.append(module.register_forward_hook(capture))
        _, fake, real_score, fake_score = _bound_scores(
            term, context, trainer.graph, 'generator', term.generator_phase, first='fake')
        if fake is not image or any(len(values) != 2 for values in captured.values()):
            raise ValueError('Expected exactly one fake and one real invocation of every multidepth score head')
        if tuple(fake_score.shape) != (len(image), 1) or real_score.shape != fake_score.shape:
            raise ValueError('Expected one scalar discriminator score per image')
        fake_heads = {name: values[0] for name, values in captured.items()}
        real_heads = {name: values[1] for name, values in captured.items()}
        for score, heads in ((fake_score, fake_heads), (real_score, real_heads)):
            reconstruction = sum(COEFFICIENTS[name] * value for name, value in heads.items())
            if not torch.allclose(score, reconstruction, rtol=2e-5, atol=2e-6):
                raise ValueError('Discriminator does not match the guarded pixel-plus-four-feature linear score mixture')
        loss = term.weight * term.gan.g_loss(fake_score, real_score)
        names = list(COEFFICIENTS)
        cotangents = torch.autograd.grad(loss, [fake_score, *[fake_heads[name] for name in names]], retain_graph=True)
        total = torch.autograd.grad(loss, image, retain_graph=True)[0].detach()
        contributions = {}
        for index, name in enumerate(names):
            cotangent = cotangents[index + 1]
            if not torch.allclose(cotangent, cotangents[0] * COEFFICIENTS[name], rtol=2e-5, atol=1e-8):
                raise ValueError('Branch cotangent disagrees with the guarded additive mixture')
            contributions[name] = torch.autograd.grad(fake_heads[name], image,
                grad_outputs=cotangent.detach(), retain_graph=index < len(names) - 1)[0].detach()
        reconstructed = sum(contributions.values())
        residual = reconstructed - total
        total_norm = _norm(total)
        relative_error = _norm(residual) / max(total_norm, 1e-30)
        # Separate branch VJPs reorder floating-point accumulation (including
        # configured TF32 kernels). Bound aggregate reconstruction error; a
        # coordinatewise relative check is unstable at cancellation zeros.
        if relative_error > 1e-3:
            raise ValueError(f'Branch image gradients do not reconstruct the actual image gradient ({relative_error:g})')
        norms = {name: _norm(value) for name, value in contributions.items()}
        norm_sum = sum(norms.values())
        paired_margin = real_score.detach() - fake_score.detach()
        branches = {}
        for name, gradient in contributions.items():
            branches[name] = {'module': modules[name][0], 'coefficient': COEFFICIENTS[name],
                'fake_score': _score_stats(fake_heads[name]), 'real_score': _score_stats(real_heads[name]),
                'loss_to_branch_score_cotangent': _score_stats(cotangents[names.index(name) + 1]),
                'image_gradient': _image_stats(gradient),
                'norm_fraction_of_sum_of_branch_norms': norms[name] / norm_sum if norm_sum else None,
                'norm_ratio_to_total_image_gradient': norms[name] / total_norm if total_norm else None,
                'cosine_with_total_image_gradient': _cosine(gradient, total)}
        result = {'schema_version': 1, 'kind': 'dinov3-multidepth-image-gradient-attribution',
            'step': trainer.step, 'batch_size': len(image), 'adversarial_term_id': term.id,
            'weighted_generator_adversarial_loss': float(loss.detach()),
            'fake_score': _score_stats(fake_score), 'real_score': _score_stats(real_score),
            'paired_real_minus_fake_score': _score_stats(paired_margin),
            'fraction_real_score_above_paired_fake': float((paired_margin > 0).float().mean()),
            'loss_to_fake_score_cotangent': _score_stats(cotangents[0]),
            'total_image_gradient': _image_stats(total), 'branches': branches,
            'pairwise_branch_gradient_cosines': [
                {'left': left, 'right': right, 'cosine': _cosine(contributions[left], contributions[right])}
                for index, left in enumerate(names) for right in names[index + 1:]],
            'cancellation_fraction': 1 - total_norm / norm_sum if norm_sum else None,
            'gradient_reconstruction': {'relative_l2_error': relative_error,
                'maximum_relative_l2_error': 1e-3, 'absolute_max_error': float(residual.abs().max())},
            'interpretation': [
                'Exact local additive branch decomposition of the configured adversarial image gradient, not a quality score.',
                'Norm fractions use the sum of branch norms; branch gradients can cancel and norm ratios to total can exceed one.',
                'Pooling measures block-average spatial energy, not semantic usefulness or a Fourier cutoff.',
                'Across-sample cosines compare fixed pixel coordinates; unaligned content can lower them without indicating a defect.',
                'No generator parameter gradients, regularizer contribution, update or independent quality claim.']}
        json.dumps(result, allow_nan=False)
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for value, original in buffers:
                if not torch.equal(value, original):
                    value.copy_(original)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        for name, state in streams.items():
            trainer.streams[name].set_state(state)
        restore_rng(rng)
    after = _digest_state(roots)
    if after != before:
        raise ValueError('Branch diagnostic changed registered model state')
    result['state_verification'] = {'before_sha256': before, 'after_sha256': after,
                                    'unchanged': True, 'optimizer_steps': 0}
    return result
