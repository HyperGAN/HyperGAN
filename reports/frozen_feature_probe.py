"""Matched online DINO feature diagnostics for the 128px multidepth testbed.

This is a research helper, not standard Inception KID or a quality/convergence
certificate. The fixed feature extractor also participates in the trained
critic. A single small bank gives a noisy distribution-distance estimate;
negative unbiased MMD estimates are valid and are deliberately not clamped.
Use an exclusively owned trainer and explicit inputs, never a live trainer
being updated concurrently.
"""
import math

import torch

from hypergan.objective_program import _bound_scores
from hypergan.signal_structure import _hash
from hypergan.startup_dynamics import _protected, _restore, _snapshot


def polynomial_mmd2_unbiased(real, fake):
    """Unbiased squared MMD with k(x,y)=(x@y / feature_width + 1)**3."""
    if (real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]
            or real.shape[1] == 0 or min(len(real), len(fake)) < 2):
        raise ValueError('Polynomial MMD requires two feature matrices with at least two rows and equal positive width')
    real, fake = real.detach().cpu().double(), fake.detach().cpu().double()
    if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
        raise ValueError('Polynomial MMD features must be finite')
    width = real.shape[1]
    rr = (real @ real.T / width + 1).pow(3)
    ff = (fake @ fake.T / width + 1).pow(3)
    rf = (real @ fake.T / width + 1).pow(3)
    value = ((rr.sum() - rr.diagonal().sum()) / (len(real) * (len(real) - 1))
             + (ff.sum() - ff.diagonal().sum()) / (len(fake) * (len(fake) - 1))
             - 2 * rf.mean())
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Polynomial MMD overflowed')
    return result


def _architecture(trainer):
    from hndl.operators.pretrained import Pretrained
    from hypergan.pretrained_providers import _dinov3_multidepth

    terms = trainer.program.adversarial_terms
    if len(terms) != 1:
        raise ValueError('Frozen-feature probe requires one multidepth discriminator objective')
    term = terms[0]
    phase = term.generator_phase
    if (phase.fake.path != 'generated' or phase.real.path != 'batch.real'
            or phase.fake.detach_sample or phase.fake.detach_score or phase.real.detach_score
            or len(term.routes) != 1 or term.routes[0].path != 'candidate'):
        raise ValueError('Frozen-feature probe requires generated/real image bindings and one candidate-only critic route')
    found = [(name, module) for name, module in term.module.named_modules()
             if isinstance(module, Pretrained) and name.rsplit('.', 1)[-1] == 'n_backbone']
    if len(found) != 1:
        raise ValueError('Expected one native pretrained n_backbone node')
    name, backbone = found[0]
    if (backbone.readout != 'multidepth'
            or getattr(backbone.provider, 'read', None) is not _dinov3_multidepth
            or backbone.source.config.get('provider') != 'dinov3_vits16'):
        raise ValueError('Expected the registered DINOv3 ViT-S/16 multidepth readout')
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError('The pretrained feature extractor must be frozen')
    if any(module.training for module in backbone.model.modules()):
        raise ValueError('The pretrained feature extractor must already be in evaluation mode')
    return term, name, backbone


def _score_stats(prefix, value):
    value = value.detach().cpu().double()
    if not torch.isfinite(value).all():
        raise ValueError('Critic scores must be finite')
    return {prefix + '_score_' + key: float(item) for key, item in {
        'mean': value.mean(), 'std': value.std(unbiased=False),
        'min': value.min(), 'max': value.max(),
    }.items()}


def measure_frozen_features(trainer, bank):
    """Return JSON-safe scalars from one online-G and two critic forwards.

    ``bank`` is ``(batch, (latent, ids))``, as accepted by ``trainer._draw``.
    To include learned-prior motion, the caller recomputes ``latent`` using the
    current prior and fixed IDs/noise. To isolate G, supply a fixed latent tensor.
    The caller also chooses a common measurement RNG state across checkpoints.
    No sampler advances because both arguments are explicit. All entry training
    state (including RNG, optimizers, EMA, gradients and caches) is restored even
    on failure, and any mutation of protected tensors fails the measurement.
    """
    if not isinstance(bank, (tuple, list)) or len(bank) != 2 or any(item is None for item in bank):
        raise ValueError('Frozen-feature probe requires an explicit batch and latent draw')
    term, backbone_name, backbone = _architecture(trainer)
    entry = _snapshot(trainer)
    protected = _protected(trainer)
    protected_before = _hash(protected)
    handle = None
    captured = []
    expected_images = []

    def capture(module, args, output):
        index = len(captured)
        if index >= 2 or len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError('Expected exactly one fake and one real backbone invocation')
        candidate = expected_images[index]
        count = len(candidate)
        value = args[0]
        if tuple(value.shape) != (2 * count, 3, 128, 128):
            raise ValueError('Expected 128px candidate-first/gray-context-second backbone input')
        mean = value.new_tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
        std = value.new_tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
        expected = (candidate.to(value) * .5 + .5 - mean) / std
        gray = ((.5 - mean) / std).expand_as(value[count:])
        if (not torch.allclose(value[:count], expected, rtol=2e-5, atol=2e-6)
                or not torch.allclose(value[count:], gray, rtol=2e-5, atol=2e-6)):
            raise ValueError('DINO input ordering, normalization or fixed-gray context changed')
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (2 * count, 1536, 8, 8):
            raise ValueError('Expected four 384-channel DINO depth maps on an 8x8 patch grid')
        # Exclude the constant context half and the first three feature depths.
        features = output[:count, -384:].detach().double().mean(dim=(-2, -1)).cpu()
        if not torch.isfinite(features).all():
            raise ValueError('DINO features must be finite')
        captured.append(features)

    try:
        with torch.no_grad():
            batch, _, context = trainer._draw(*bank)
            fake = context['generated']
            real = batch['real']
            if tuple(fake.shape[1:]) != (3, 128, 128) or fake.shape != real.shape or len(fake) < 2:
                raise ValueError('Frozen-feature probe requires at least two matched 128px RGB images')
            expected_images.extend((fake, real))
            handle = backbone.register_forward_hook(capture)
            _, _, real_score, fake_score = _bound_scores(
                term, context, trainer.graph, 'generator', term.generator_phase, first='fake')
            if len(captured) != 2:
                raise ValueError('Expected exactly two backbone invocations')
            if tuple(real_score.shape) != (len(real), 1) or fake_score.shape != real_score.shape:
                raise ValueError('Expected one scalar critic score per image')
            fake_features, real_features = captured
            real_spread = float(real_features.var(dim=0, unbiased=False).mean().sqrt())
            fake_spread = float(fake_features.var(dim=0, unbiased=False).mean().sqrt())
            g_loss = float(term.weight * term.gan.g_loss(fake_score, real_score))
            d_loss = float(term.weight * term.gan.d_loss(real_score, fake_score))
            if not math.isfinite(g_loss) or not math.isfinite(d_loss):
                raise ValueError('Matched adversarial losses must be finite')
            result = {
                'protocol': 'online_dinov3_block11_spatial_mean_candidate_only_poly3',
                'backbone_path': backbone_name,
                'samples': len(real), 'feature_width': 384,
                'dino_poly3_mmd2_unbiased': polynomial_mmd2_unbiased(real_features, fake_features),
                'dino_feature_mean_distance_rms': float((real_features.mean(0) - fake_features.mean(0)).square().mean().sqrt()),
                'dino_real_feature_spread': real_spread,
                'dino_fake_feature_spread': fake_spread,
                'dino_feature_spread_ratio': fake_spread / real_spread if real_spread else None,
                **_score_stats('real', real_score), **_score_stats('fake', fake_score),
                'paired_real_minus_fake_score_mean': float((real_score - fake_score).double().mean()),
                'matched_generator_adversarial_loss': g_loss,
                # This uses generator-phase draws, with no lazy penalty or prior
                # regularizer; it is not the separately sampled training D loss.
                'matched_discriminator_adversarial_loss': d_loss,
                'generator_forwards': 1, 'critic_forwards': 2, 'pretrained_forwards': 2,
                'pretrained_images_including_gray_context': 4 * len(real),
                'protected_state_sha256': protected_before,
            }
    finally:
        if handle is not None:
            handle.remove()
        try:
            changed = _hash(protected) != protected_before
        finally:
            _restore(trainer, entry)
        if _hash(_protected(trainer)) != protected_before:
            raise RuntimeError('Frozen-feature probe failed to restore protected state')
        if changed:
            raise RuntimeError('Frozen-feature forward mutated protected state; entry state was restored')
    return result
