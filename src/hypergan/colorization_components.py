"""Conditional 256px logo models with ParticleGAN's hard fixed-sigma posterior.

The posterior matches its selected prior component exactly; its joint KL is the
constant log(K). The straight-through routing derivative is a surrogate. There
are no spatial generator skips and no learned posterior offset or variance.
"""
import importlib
import math
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .image_components import SAGANAttention, _verified_file


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class _UpsampleBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _norm(cout), nn.LeakyReLU(.2),
            nn.Conv2d(cout, cout, 3, padding=1), _norm(cout), nn.LeakyReLU(.2))
        self.skip = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return (self.layers(x) + self.skip(x)) / math.sqrt(2)


class ColorizationGenerator(nn.Module):
    """Compact latent-only RGB decoder with SAGAN attention at 16 or 32 pixels."""
    def __init__(self, z_dim=128, width=32, attention_size=16):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        _positive_integer(width, 'width')
        if attention_size not in (16, 32):
            raise ValueError('Generator attention_size must be 16 or 32')
        self.attention_size = attention_size
        channels = [8 * width, 8 * width, 4 * width, 2 * width, width, width, width]
        self.input = nn.Linear(z_dim, channels[0] * 16)
        self.input_norm = _norm(channels[0])
        self.blocks = nn.ModuleList(_UpsampleBlock(a, b) for a, b in zip(channels, channels[1:]))
        self.attention = SAGANAttention(channels[int(math.log2(attention_size)) - 2])
        self.output = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f'Generator requires z [batch,{self.z_dim}]')
        h = F.leaky_relu(self.input_norm(self.input(z).reshape(len(z), -1, 4, 4)), .2)
        for block in self.blocks:
            h = block(h)
            if h.shape[-1] == self.attention_size:
                h = self.attention(h)
        return self.output(h).tanh()


class GrayscaleRoutingEncoder(nn.Module):
    """Encode grayscale, select one particle and add the prior's fixed sigma noise.

    ``latent`` sends adversarial gradients to the selected prior means as well as
    the encoder. ``reconstruction_latent`` uses identical values/noise but trains
    only the encoder when decoded through a parameter-frozen generator. Routing
    uses detached means and a soft backward surrogate; particle updates come from
    selected centers. Noise uses Torch's checkpointed global RNG.
    """
    def __init__(self, z_dim=128, width=32, temperature=.125, detach_means=False):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        _positive_integer(width, 'width')
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Routing temperature must be positive and finite')
        if type(detach_means) is not bool:
            raise ValueError('detach_means must be a boolean')
        self.temperature, self.detach_means = temperature, detach_means
        channels = [1, width, 2 * width, 4 * width, 8 * width, 8 * width, 8 * width]
        layers = []
        for a, b in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(a, b, 4, stride=2, padding=1), _norm(b), nn.LeakyReLU(.2)])
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.query = nn.Linear(channels[-1] * 16, z_dim)

    def forward(self, gray, means, sigma):
        if gray.ndim != 4 or tuple(gray.shape[1:]) != (1, 256, 256):
            raise ValueError('Encoder requires gray [batch,1,256,256] in [-1,1]')
        if means.ndim != 2 or means.shape[1] != self.z_dim or means.shape[0] < 1:
            raise ValueError(f'Encoder requires means [particles,{self.z_dim}]')
        if not isinstance(sigma, torch.Tensor) or sigma.numel() != 1:
            raise ValueError('Encoder requires the prior fixed scalar sigma tensor')
        if means.device != gray.device or sigma.device != gray.device:
            raise ValueError('Encoder gray, prior means and sigma must share a device')
        torch._assert_async(torch.isfinite(sigma).all() & (sigma > 0).all(),
                            'Hard VAE posterior requires positive finite fixed sigma')
        query = self.query(self.features(gray))
        query = F.layer_norm(query, (self.z_dim,))
        fixed = means.detach()
        distances = (query.square().sum(1, keepdim=True) + fixed.square().sum(1)[None]
                     - 2 * query @ fixed.T) / self.z_dim
        ids = distances.argmin(1)
        soft = (-distances / self.temperature).softmax(1)
        proxy = soft @ fixed
        surrogate = proxy - proxy.detach()
        noise = sigma.detach().reshape(()) * torch.randn_like(query)
        reconstruction = fixed[ids] + surrogate + noise
        center = fixed[ids] if self.detach_means else means[ids]
        return {'latent': center + surrogate + noise, 'reconstruction_latent': reconstruction,
                'ids': ids, 'soft': soft}


def _load_dinov3(source_path, source_commit, weights_path, weights_sha256):
    """Load an explicitly pinned clean local upstream checkout; never download."""
    source = Path(source_path).expanduser().resolve()
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('DINOv3 source_commit must be a full lowercase Git SHA')
    if not (source / 'dinov3' / 'hub' / 'backbones.py').is_file():
        raise FileNotFoundError(f'DINOv3 source checkout missing: {source}')
    try:
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f'Cannot verify local DINOv3 Git checkout: {source}') from exc
    if actual != source_commit or dirty:
        raise ValueError(f'DINOv3 source must be clean at {source_commit}; got {actual}, dirty={bool(dirty)}')
    weights = _verified_file(weights_path, weights_sha256)
    # Explicit local import instead of Torch Hub's network/cache discovery.
    sys.path.insert(0, str(source))
    try:
        package = importlib.import_module('dinov3')
        if Path(package.__file__).resolve().parent != source / 'dinov3':
            raise ValueError('A different DINOv3 package is imported; use the configured pinned checkout')
        backbones = importlib.import_module('dinov3.hub.backbones')
        backbone = backbones.dinov3_vits16(pretrained=False)
    finally:
        sys.path.remove(str(source))
    backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return backbone


class _PixelHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        channels = [4, width, width * 2, width * 4, width * 4, width * 4, width * 4]
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Conv2d(a, b, 4, stride=2, padding=1), nn.LeakyReLU(.2))
            for a, b in zip(channels, channels[1:]))
        self.attention = SAGANAttention(width * 4)
        self.output = nn.Linear(width * 4 * 16, 1)

    def forward(self, x, gray):
        h = torch.cat([x, gray], 1)
        for block in self.blocks:
            h = block(h)
            if h.shape[-1] == 16:
                h = self.attention(h)
        return self.output(h.flatten(1))


class DINOv3Discriminator(nn.Module):
    """Frozen ViT-S/16 image features with trainable attention and grayscale pixel head.

    The backbone always stays in evaluation mode, but image derivatives flow
    through it. Math SDPA supports the double backward required by b-cap; Flash
    and efficient SDPA do not provide this derivative on supported Torch builds.
    """
    def __init__(self, source_path, source_commit, weights_path, weights_sha256,
                 width=32, feature_width=64):
        super().__init__()
        _positive_integer(width, 'width')
        _positive_integer(feature_width, 'feature_width')
        self.backbone = _load_dinov3(source_path, source_commit, weights_path, weights_sha256)
        self.backbone.eval().requires_grad_(False)
        self.pixel = _PixelHead(width)
        self.feature_project = nn.Sequential(nn.Conv2d(384, feature_width, 1),
                                             _norm(feature_width), nn.LeakyReLU(.2))
        self.attention = SAGANAttention(feature_width)
        self.feature_output = nn.Linear(feature_width * 16, 1)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.pretrained_metadata = {'architecture': 'dinov3_vits16', 'pretraining': 'LVD-1689M',
                                    'source_commit': source_commit, 'weights_sha256': weights_sha256,
                                    'input_size': 256, 'sdpa_backend': 'math'}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.backbone.requires_grad_(False)
        return self

    def forward(self, x, gray):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Discriminator requires x [batch,3,256,256] in [-1,1]')
        if gray.shape != (len(x), 1, 256, 256):
            raise ValueError('Discriminator requires matching gray [batch,1,256,256]')
        normalized = (x * .5 + .5 - self.mean) / self.std
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            tokens = self.backbone.forward_features(normalized)['x_norm_patchtokens']
        if tokens.shape != (len(x), 256, 384):
            raise ValueError('DINOv3 ViT-S/16 must return 256 patch tokens of width 384')
        features = tokens.transpose(1, 2).reshape(len(x), 384, 16, 16)
        features = self.attention(self.feature_project(features))
        # Nonoverlapping reduction avoids adaptive-pool CUDA backward atomics.
        pooled = features.reshape(len(x), features.shape[1], 4, 4, 4, 4).mean((3, 5))
        logits = self.feature_output(pooled.flatten(1))
        return (self.pixel(x, gray) + logits) / math.sqrt(2)
