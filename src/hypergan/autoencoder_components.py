"""RGB particle AE encoder for the 256px logo experiment.

Routing follows ParticleGAN's particle_ae, with the image recipe's normalized
query, mean distance and .125 temperature. Reconstruction trains E, G and prior.
"""
import math

import torch
from torch import nn

from .colorization_components import _network, _width_parameters, _positive_integer


class ParticleAEEncoder256(nn.Module):
    """RGB -> nearest particle + sigma * 3*tanh(offset/3), without noise or KL.

    The public graph supplies prior means and sigma as tensors. This implements
    ParticleGAN's tensor routing formula without owning a second prior. Selected
    means receive gradients; query gradients use its soft straight-through rule.
    """
    def __init__(self, z_dim=128, width=32, temperature=.125, networks=None):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        _positive_integer(width, 'width')
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Routing temperature must be positive and finite')
        self.temperature = temperature
        hidden = 8 * width * 16
        self.features = _network('colorization_encoder_features', (3, 256, 256),
                                 (hidden,), _width_parameters(width), networks)
        self.query = _network('colorization_query', (hidden,), (z_dim,),
                              {'z_dim': z_dim}, networks)
        self.offset = _network('autoencoder_offset', (hidden,), (z_dim,),
                               {'z_dim': z_dim}, networks)

    def forward(self, x, means, sigma):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Encoder requires RGB x [batch,3,256,256] in [-1,1]')
        if means.ndim != 2 or means.shape[1] != self.z_dim or means.shape[0] < 2:
            raise ValueError(f'Encoder requires means [particles>=2,{self.z_dim}]')
        if means.device != x.device or sigma.device != x.device or means.dtype != x.dtype:
            raise ValueError('Encoder input and prior must share device and floating dtype')
        if sigma.numel() != 1:
            raise ValueError('AE prior sigma must be scalar')
        torch._assert_async(torch.isfinite(sigma).all() & (sigma > 0).all(),
                            'AE requires positive finite fixed sigma')
        h = self.features(x)
        query = self.query(h)
        fixed = means.detach()
        distance = (query.square().sum(1, keepdim=True) + fixed.square().sum(1)[None]
                    - 2 * query @ fixed.T) / self.z_dim
        log_probs = (-distance / self.temperature).log_softmax(1)
        ids = log_probs.argmax(1)
        soft = log_probs.exp()
        proxy = soft @ fixed
        center = means[ids] + (proxy - proxy.detach())
        offset = torch.tanh(self.offset(h) / 3)
        # Preserve particle_ae's multiplication order as well as its derivative.
        return {'latent': center + sigma * 3 * offset, 'ids': ids, 'offset': 3 * offset,
                'soft': soft}
