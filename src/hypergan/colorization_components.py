"""Conditional 256px logo models with ParticleGAN's hard fixed-sigma posterior.

The posterior matches its selected prior component exactly; its joint KL is the
constant log(K). The straight-through routing derivative is a surrogate. There
are no spatial generator skips and no learned posterior offset or variance.
"""
import math

import torch
from torch import nn

from .image_components import SAGANAttention, _verified_file
from .hndl_networks import build_network
from .network_config import SourceFragment
from .image_components import attention_source


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _width_parameters(width):
    return {**{f'w{n}': n * width for n in (1, 2, 4, 8)},
            **{f'g{n}': math.gcd(8, n * width) for n in (1, 2, 4, 8)},
            'w8x16': 8 * width * 16}


def _network(stem, input_shape, output_shape, parameters=None, networks=None):
    return build_network(source=(networks or {}).get(stem), file=stem + '.hndl',
                         input_shape=({key: ('B', *shape) for key, shape in input_shape.items()}
                                      if isinstance(input_shape, dict) else ('B', *input_shape)),
                         output_shape=({key: ('B', *shape) for key, shape in output_shape.items()}
                                       if isinstance(output_shape, dict) else ('B', *output_shape)),
                         parameters=parameters)


def _attention(channels, size, networks=None):
    return SourceFragment(attention_source(channels, size, prefix='attention', input='attention_input',
                            source=(networks or {}).get('image_attention')))


class ColorizationGenerator(nn.Module):
    """Compact latent-only RGB decoder with SAGAN attention at 16 or 32 pixels."""
    def __init__(self, z_dim=128, width=32, attention_size=16, networks=None):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        _positive_integer(width, 'width')
        if attention_size not in (16, 32):
            raise ValueError('Generator attention_size must be 16 or 32')
        self.attention_size = attention_size
        parameters = _width_parameters(width)
        parameters['attention'] = _attention((4 if attention_size == 16 else 2) * width,
                                              attention_size, networks)
        self.network = _network(f'colorization_generator{attention_size}', (z_dim,),
                                (3, 256, 256), parameters, networks)

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f'Generator requires z [batch,{self.z_dim}]')
        return self.network(z)


class GrayscaleImage(nn.Module):
    """Differentiable luminance, matching the colorization data transform."""
    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('GrayscaleImage requires RGB [batch,3,height,width]')
        return x[:, 0:1] * .299 + x[:, 1:2] * .587 + x[:, 2:3] * .114


class GrayscaleRoutingEncoder(nn.Module):
    """Encode grayscale, select one particle and add the prior's fixed sigma noise.

    ``latent`` sends adversarial gradients to the selected prior means as well as
    the encoder. ``reconstruction_latent`` uses identical values/noise but trains
    only the encoder when decoded through a parameter-frozen generator. Routing
    uses detached means and a soft backward surrogate; particle updates come from
    selected centers. Noise uses Torch's checkpointed global RNG.
    """
    def __init__(self, z_dim=128, width=32, temperature=.125, detach_means=False, networks=None):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        _positive_integer(width, 'width')
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Routing temperature must be positive and finite')
        if type(detach_means) is not bool:
            raise ValueError('detach_means must be a boolean')
        self.temperature, self.detach_means = temperature, detach_means
        hidden = 8 * width * 16
        self.features = _network('colorization_encoder_features', (1, 256, 256),
                                 (hidden,), _width_parameters(width), networks)
        self.query = _network('colorization_query', (hidden,), (z_dim,),
                              {'z_dim': z_dim}, networks)

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


def _load_dinov3(source_path, source_commit, weights_path, weights_sha256,
                 *, multidepth=False, networks=None):
    """Build a pinned native pretrained node; HNDL owns checkpoint loading."""
    from .pretrained_providers import dinov3_registry
    registry = dinov3_registry(source_path, source_commit)
    weights = _verified_file(weights_path, weights_sha256)
    stem = 'colorization_dinov3_' + ('multidepth' if multidepth else 'patch_tokens')
    output_shape = ({f'm{i}': ('B', 384, 16, 16) for i in range(4)} if multidepth
                    else ('B', 384, 16, 16))
    return build_network(source=(networks or {}).get(stem), file=stem + '.hndl',
                         input_shape=('B', 3, 256, 256), output_shape=output_shape,
                         parameters={'weights_path': str(weights), 'weights_sha256': weights_sha256},
                         registry=registry)


class _PixelHead(nn.Module):
    def __init__(self, width, networks=None):
        super().__init__()
        parameters = _width_parameters(width)
        parameters['attention'] = _attention(width * 4, 16, networks)
        self.network = _network('colorization_pixel_head', {'rgb': (3, 256, 256),
                                                          'gray': (1, 256, 256)}, (1,),
                                parameters, networks)

    def forward(self, x, gray):
        return self.network(rgb=x, gray=gray)


class DINOv3Discriminator(nn.Module):
    """Frozen ViT-S/16 image features with trainable attention and grayscale pixel head.

    The backbone always stays in evaluation mode, but image derivatives flow
    through it. Math SDPA supports the double backward required by b-cap; Flash
    and efficient SDPA do not provide this derivative on supported Torch builds.
    """
    def __init__(self, source_path, source_commit, weights_path, weights_sha256,
                 width=32, feature_width=64, networks=None):
        super().__init__()
        _positive_integer(width, 'width')
        _positive_integer(feature_width, 'feature_width')
        self.backbone = _load_dinov3(source_path, source_commit, weights_path, weights_sha256,
                                    networks=networks)
        self.pixel = _PixelHead(width, networks)
        self.feature_project = _network('colorization_feature_project', (384, 16, 16),
                                        (feature_width, 16, 16),
                                        {'feature_width': feature_width,
                                         'groups': math.gcd(8, feature_width)}, networks)
        self.attention = SAGANAttention(feature_width, image_size=16, networks=networks)
        self.feature_output = _network('colorization_linear_head', (feature_width, 16, 16),
                                       (1,), networks=networks)
        self.score = _network('colorization_joint_score', {'pixel': (1,), 'feature': (1,)},
                               (1,), networks=networks)
        self.pretrained_metadata = {'architecture': 'dinov3_vits16', 'pretraining': 'LVD-1689M',
                                    'source_commit': source_commit, 'weights_sha256': weights_sha256,
                                    'input_size': 256, 'sdpa_backend': 'math'}

    def forward(self, x, gray):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Discriminator requires x [batch,3,256,256] in [-1,1]')
        if gray.shape != (len(x), 1, 256, 256):
            raise ValueError('Discriminator requires matching gray [batch,1,256,256]')
        features = self.backbone(x)
        features = self.attention(self.feature_project(features))
        logits = self.feature_output(features)
        return self.score(pixel=self.pixel(x, gray), feature=logits)


class DINOv3ProjectedDiscriminator(nn.Module):
    """One unconditional frozen-feature path with random projection and attention.

    Candidate RGB -> frozen DINOv3 -> frozen random 1x1 channel mixing and
    3x3 local spatial mixing -> trainable attention -> trainable scalar head.
    This follows Projected GAN's fixed random feature-projection principle
    (https://github.com/autonomousvision/projected-gan/blob/main/pg_modules/projector.py).
    It is a single final-feature-map adaptation: the 3x3 convolution mixes
    neighboring patches, not multiple feature scales. It does not reproduce
    the paper's multiscale cross-scale mixing or separate scale discriminators.
    ``head='conv'`` adds a nonlinear, spectral-normalized discriminator over
    the projected patches (16 -> 8 -> 4 -> 1). Both head variants are resolved from HNDL configuration.
    With the convolutional head, ``pixel_width > 0`` concatenates a learned RGB
    stem with projected DINO features before the shared attention and head.
    This preserves one image input, one backbone call and one scalar critic.

    Projection initialization uses Torch's checkpointed global RNG, and all
    frozen weights are included in the module state. Image derivatives remain
    enabled through both frozen modules; math SDPA permits b-cap double backward.
    The original two-path DINOv3Discriminator remains available for prior runs.
    """
    def __init__(self, source_path, source_commit, weights_path, weights_sha256,
                 feature_width=64, head='linear', pixel_width=0, networks=None):
        super().__init__()
        _positive_integer(feature_width, 'feature_width')
        if head not in ('linear', 'conv'):
            raise ValueError("Projected discriminator head must be 'linear' or 'conv'")
        if type(pixel_width) is not int or pixel_width < 0:
            raise ValueError('pixel_width must be a nonnegative integer')
        if pixel_width and head != 'conv':
            raise ValueError("pixel_width > 0 requires head='conv'")
        self.head = head
        self.pixel_width = pixel_width
        self.backbone = _load_dinov3(source_path, source_commit, weights_path, weights_sha256,
                                    networks=networks)
        self.feature_project = _network('colorization_random_project', (384, 16, 16),
                                        (feature_width, 16, 16),
                                        {'feature_width': feature_width}, networks)
        feature_channels = feature_width
        if pixel_width:
            self.pixel_features = _network('colorization_pixel_features', (3, 256, 256),
                                           (feature_width, 16, 16),
                                           {'pixel_width': pixel_width, 'p2': 2 * pixel_width,
                                            'feature_width': feature_width}, networks)
            self.feature_concat = _network('colorization_feature_concat',
                                            {'features': (feature_width, 16, 16),
                                             'pixels': (feature_width, 16, 16)},
                                            (2 * feature_width, 16, 16), networks=networks)
            feature_channels += feature_width
        self.attention = SAGANAttention(feature_channels, image_size=16, networks=networks)
        self.feature_output = _network('colorization_' + head + '_head',
                                       (feature_channels, 16, 16), (1,),
                                       {'f2': 2 * feature_width, 'f4': 4 * feature_width}, networks)
        self.pretrained_metadata = {'architecture': 'dinov3_vits16', 'pretraining': 'LVD-1689M',
                                    'source_commit': source_commit, 'weights_sha256': weights_sha256,
                                    'input_size': 256, 'sdpa_backend': 'math',
                                    'projection': 'frozen_random_1x1_channel_3x3_spatial',
                                    'feature_scales': 1, 'head': head, 'pixel_width': pixel_width}

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Discriminator requires x [batch,3,256,256] in [-1,1]')
        features = self.backbone(x)
        features = self.feature_project(features)
        if self.pixel_width:
            features = self.feature_concat(features=features, pixels=self.pixel_features(x))
        features = self.attention(features)
        return self.feature_output(features)


class _MultiDepthProjection(nn.Module):
    """Bind four depth maps to a native HNDL multi-input/output graph."""
    sizes = (32, 16, 8, 4)

    def __init__(self, width, networks=None):
        super().__init__()
        self.width = width
        self.network = _network('colorization_multidepth_projection',
                                {f'm{i}': (384, 16, 16) for i in range(4)},
                                {f'o{i}': (width, size, size) for i, size in enumerate(self.sizes)},
                                {'width': width}, networks)

    def forward(self, maps):
        outputs = self.network(**{f'm{i}': value for i, value in enumerate(maps)})
        return [outputs[f'o{i}'] for i in range(4)]


class _ProjectedScaleHead(nn.Module):
    def __init__(self, width, size, networks=None):
        super().__init__()
        parameters = _width_parameters(width)
        parameters['attention'] = _attention(width * (2 if size == 32 else 1),
                                              min(size, 16), networks)
        self.layers = _network(f'colorization_scale_head{size}', (width, size, size),
                               (1,), parameters, networks)

    def forward(self, x):
        return self.layers(x)


class DINOv3MultiScaleDiscriminator(nn.Module):
    """One DINOv3 pass, four frozen projected depths, and learned scale critics.

    This adapts Projected GAN's channel mixing, cross-scale mixing and separate
    convolutional critics to ViT-S/16. Blocks 2, 5, 8 and 11 all return 16x16
    patch maps; nearest upsampling and nonoverlapping averaging construct a
    synthetic 32/16/8/4 pyramid. These are not native backbone spatial scales.
    Four scalar logits are averaged, unlike upstream's concatenated patch
    logits. Each trainable head includes attention at at most 16x16.

    Backbone and random channel/fusion convolutions remain frozen and in eval
    mode. Image derivatives pass through them. Math SDPA and linear resampling
    retain input double backward for b-cap, including deterministic execution.
    """
    blocks = (2, 5, 8, 11)

    def __init__(self, source_path, source_commit, weights_path, weights_sha256,
                 feature_width=64, networks=None):
        super().__init__()
        _positive_integer(feature_width, 'feature_width')
        self.backbone = _load_dinov3(source_path, source_commit, weights_path, weights_sha256,
                                    multidepth=True, networks=networks)
        self.feature_project = _MultiDepthProjection(feature_width, networks)
        self.heads = nn.ModuleList(_ProjectedScaleHead(feature_width, size, networks)
                                  for size in self.feature_project.sizes)
        self.score = _network('colorization_multiscale_score',
                               {f'scale{i}': (1,) for i in range(4)}, (1,), networks=networks)
        self.pretrained_metadata = {
            'architecture': 'dinov3_vits16', 'pretraining': 'LVD-1689M',
            'source_commit': source_commit, 'weights_sha256': weights_sha256,
            'input_size': 256, 'sdpa_backend': 'math',
            'projection': 'frozen_random_channel_and_topdown_fusion',
            'feature_blocks': list(self.blocks), 'feature_scales': 4,
            'native_feature_sizes': [16, 16, 16, 16],
            'projected_feature_sizes': list(self.feature_project.sizes),
            'resampling': 'nearest_upsample_nonoverlapping_mean_downsample',
            'head': 'spectral_conv_attention', 'aggregation': 'mean_scalar_logits',
        }

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Discriminator requires x [batch,3,256,256] in [-1,1]')
        maps = self.backbone(x)
        projected = self.feature_project([maps[f'm{i}'] for i in range(4)])
        return self.score(**{f'scale{i}': head(feature)
                             for i, (head, feature) in enumerate(zip(self.heads, projected))})


class DCGANDiscriminator256(nn.Module):
    """Unconditional RGB pixel critic for a 256px discriminator control.

    Six stride-two convolutions reduce 256px to 4px before a scalar head.
    Spectral normalization bounds each learned layer; batch normalization is
    deliberately absent so a sample's score has no dependence on its peers.
    Unlike the projected critic, this model has no pretrained feature path.
    """
    def __init__(self, width=32, spectral_norm=True, networks=None):
        super().__init__()
        _positive_integer(width, 'width')
        if type(spectral_norm) is not bool:
            raise ValueError('spectral_norm must be a boolean')

        parameters = _width_parameters(width)
        parameters['spectral_norm'] = spectral_norm
        self.network = _network('colorization_dcgan256', (3, 256, 256), (1,),
                                parameters, networks)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Discriminator requires x [batch,3,256,256] in [-1,1]')
        return self.network(x)
