"""HyperGAN's differentiable image augmentation as a native HNDL operator."""
from hndl import Arg, operator

from .diff_augment import DiffAugment


@operator(
    'diff_augment',
    identity='hypergan.diff_augment',
    summary='Apply differentiable color, translation and cutout augmentation while training.',
    shape='x[B, C, H, W] -> out[B, C, H, W]',
    args={'transforms': Arg(str, 'color,translation,cutout',
                            help='Comma-separated DiffAugment transforms; empty disables augmentation.')},
    category='regularization',
)
class HNDLDiffAugment(DiffAugment):
    """Keep augmentation in the graph, before feature normalization.

    ``transforms`` names the DiffAugment policy because ``policy`` is reserved
    for HNDL construction metadata. Freezing parameters leaves augmentation
    enabled during generator updates; ``eval()`` disables it without RNG use.
    """

    def __init__(self, transforms='color,translation,cutout'):
        super().__init__(policy=transforms)


def register_augmentation(registry):
    """Add the host operator to one registry without modifying global state."""
    if 'diff_augment' not in registry.aliases:
        registry.add(HNDLDiffAugment)
    return registry
