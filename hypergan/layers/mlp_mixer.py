import torch.nn as nn
from hypergan.layer_shape import LayerShape
from functools import partial
import hypergan as hg
from einops.layers.torch import Rearrange, Reduce
import torch

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(x) + x


def FeedForward(dim, expansion_factor = 4, dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.PReLU(),
        dense(inner_dim, dim)
    )
class MlpMixer(hg.Layer):
    """ MLP Mixer https://arxiv.org/pdf/2105.01601.pdf"""
    def __init__(self, component, args, options):
        super(MlpMixer, self).__init__(component, args, options)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        patch_size = options.patch_size
        dims = component.current_size.dims
        image_size = component.gan.width()
        num_patches = (image_size // patch_size) ** 2
        depth = options.depth
        dim = options.dim
        expansion_factor = options.expansion_factor or 4
        num_classes = args[0]
        self.size = LayerShape(num_classes)
        channels = dims[0]
        self.net = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, chan_last))
            ) for _ in range(depth)],
            #nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def output_size(self):
        return self.size

    def forward(self, x, context):
        return self.net(x)
