import hyperchamber as hc
import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import inspect
import os

from .base_discriminator import BaseDiscriminator

#from mlp_mixer_pytorch import MLPMixer
from mlp_mixer_pytorch.mlp_mixer_pytorch import FeedForward

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(x) + x


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class MLPMixerDiscriminator(BaseDiscriminator):

    def required(self):
        return []

    def create(self):
        ops = []
        image_size = self.gan.width()
        channels = self.gan.channels()
        if self.config.conv_layers:
            ops.append(nn.Sequential(
                    nn.Conv2d(3, 64, 3, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2, 1),
                    nn.ReLU()))
            image_size = self.gan.width()//4
            channels = 128

        ops.append(MLPMixer(
                image_size = image_size,
                channels = channels,
                patch_size = (self.config.patch_size or 8),
                dim=(self.config.dim or 512),
                depth=(self.config.depth or 12),
                num_classes=(self.config.num_classes or 1)))
        self.net = nn.Sequential(*ops)
    def forward(self, x):
        return self.net(x).view(self.gan.batch_size(), -1)
