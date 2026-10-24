"""HyperGAN's differentiable image augmentation as a native HNDL operator."""
from hndl import Arg, operator

from .diff_augment import DiffAugment


@operator(
    'diff_augment',
    identity='hypergan.diff_augment',
    summary='Apply differentiable color, translation and cutout augmentation while training.',
    shape='x[B, C, H, W] -> out[B, C, H, W]',
    args={
        'transforms': Arg(str, 'color,translation,cutout',
                          help='Comma-separated DiffAugment transforms; empty disables augmentation.'),
        'translation_ratio': Arg(float, 0.125, min=0, max=1,
                                 help='Maximum translation as a fraction of each image dimension.'),
        'cutout_probability': Arg(float, 1.0, min=0, max=1,
                                  help='Probability of applying cutout to the batch.'),
    },
    category='regularization',
)
class HNDLDiffAugment(DiffAugment):
    """Keep augmentation in the graph, before feature normalization.

    ``transforms`` names the DiffAugment policy because ``policy`` is reserved
    for HNDL construction metadata. Freezing parameters leaves augmentation
    enabled during generator updates; ``eval()`` disables it without RNG use.
    """

    def __init__(self, transforms='color,translation,cutout', translation_ratio=0.125,
                 cutout_probability=1.0):
        super().__init__(policy=transforms, translation_ratio=translation_ratio,
                         cutout_probability=cutout_probability)


@operator(
    'contrastive_log_odds',
    identity='hypergan.contrastive_log_odds',
    summary='Compute one-vs-rest log-odds margin for a target option index.',
    shape='x[B, K] -> out[B, 1]',
    args={
        'target_index': Arg(int, 0, min=0,
                            help='Option index of the positive hypothesis against all other options.'),
    },
    category='activation',
)
class HNDLContrastiveLogOdds(DiffAugment.__mro__[1]):
    """Project K option logits to the sufficient 1D Bayes log-odds margin."""

    def __init__(self, target_index=0):
        super().__init__()
        self.target_index = int(target_index)

    def forward(self, x):
        import torch
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError('contrastive_log_odds expects a 2D tensor [B, K] with K >= 2')
        idx = self.target_index
        if idx < 0 or idx >= x.shape[1]:
            raise ValueError(f'target_index {idx} out of bounds for K={x.shape[1]}')
        pos = x[:, idx : idx + 1]
        neg = torch.cat([x[:, :idx], x[:, idx + 1 :]], dim=-1)
        return pos - torch.logsumexp(neg, dim=-1, keepdim=True)


def register_augmentation(registry):
    """Add the host operator to one registry without modifying global state."""
    if 'diff_augment' not in registry.aliases:
        registry.add(HNDLDiffAugment)
    if 'contrastive_log_odds' not in registry.aliases:
        registry.add(HNDLContrastiveLogOdds)
    return registry
