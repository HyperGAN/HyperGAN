"""CIFAR models whose trainable and pretrained architecture lives in HNDL files.

Python adapts training inputs, validates local artifacts, and routes mixture
components. Network layers and connectivity are resolved from configuration.
"""
import hashlib
import math
from importlib.resources import files
from pathlib import Path
from string import Template

import torch
from torch import nn
from torch.nn import functional as F
from particlegan import ucd_scores

from .hndl_networks import build_network
from .network_config import SourceFragment

RESNET18_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
FEATURE_SHA256 = '5de287ab28d569dfc53a5bca4a646d4416621da29e71e80859e6117c7f90b0ac'


def _verified_file(path, expected):
    path = Path(path).expanduser()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Expected a lowercase SHA256 digest')
    if not path.is_file():
        raise FileNotFoundError(f'Required local artifact is missing: {path}; no automatic downloads')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f'Local artifact SHA256 mismatch: {path}; expected {expected}, got {actual}')
    return path


def _source(stem, networks=None, **parameters):
    source = (networks or {}).get(stem)
    if source is None:
        source = files('hypergan').joinpath('networks', stem + '.hndl').read_text()
    return Template(source).substitute(parameters)


def attention_source(channels, size, prefix='attention', input='x', source=None):
    """Render the reusable HNDL attention fragment, ending at ``prefix_out``."""
    networks = {'image_attention': source} if source is not None else None
    return SourceFragment(_source('image_attention', networks, prefix=prefix, channels=channels,
                   query_channels=max(1, channels // 8), value_channels=max(1, channels // 2),
                   size=size, positions=size * size, input=input))


class SAGANAttention(nn.Module):
    """HNDL's explicit unscaled single-head spatial attention."""
    def __init__(self, channels, image_size=4, networks=None):
        super().__init__()
        self.network = build_network(attention_source(channels, image_size,
            source=(networks or {}).get('image_attention')),
            input_shape=('B', channels, image_size, image_size),
            output_shape=('B', channels, image_size, image_size))

    @property
    def query(self):
        return self.network['attention_query']

    def forward(self, x):
        return self.network(x)


class CIFARGenerator(nn.Module):
    """Source deconvolutional G with 16x16 SAGAN attention."""
    def __init__(self, z_dim=64, width=32, attention_seed=124003, networks=None):
        super().__init__()
        source = _source('image_generator', networks, attention=attention_source(64, 16, input='h',
                         source=(networks or {}).get('image_attention')))
        self.network = build_network(source, input_shape=('B', z_dim),
            output_shape=('B', 3, 32, 32))
        # Preserve the independent attention initialization stream. The graph
        # owns its parameters; only initialization policy is applied here.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(attention_seed)
            for node in self.network.plan.nodes:
                if node.id.startswith('attention_'):
                    module = self.network[node.id]
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    def forward(self, z):
        return self.network(z)


class _PixelDiscriminator(nn.Module):
    def __init__(self, width, image_size=32, attention_seed=124003, networks=None):
        super().__init__()
        channels = [width, 2 * width] + [4 * width] * (int(math.log2(image_size)) - 3)
        blocks = []
        size = image_size
        for index, (cin, cout) in enumerate(zip(channels, channels[1:])):
            prefix = f'block{index}'
            size //= 2
            attention = ''
            if size == 16:
                attention = attention_source(cout, size, input='h',
                    source=(networks or {}).get('image_attention')) + '\nh = attention_out\n'
            skip = ('h' if cin == cout else
                    f'conv(h, {cout}, kernel_size=1, name="{prefix}_skip")')
            blocks.append(_source('image_pixel_block', networks, prefix=prefix,
                groups_in=min(8, cin), groups_out=min(8, cout), channels=cout,
                skip=skip, attention=attention))
        source = _source('image_pixel', networks, embedding=4 * width,
                         width=width, blocks='\n'.join(blocks))
        self.network = build_network(source,
            input_shape={name: ('B', 3, image_size, image_size) for name in ('image', 'context')},
            output_shape=('B', 1))
        self.block_count = len(blocks)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(attention_seed)
            SAGANAttention(64, 16)  # G's attention precedes D's attention in the source.
            for node in self.network.plan.nodes:
                if node.id.startswith('attention_'):
                    module = self.network[node.id]
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    @property
    def input(self):
        return self.network['input']

    @property
    def attention(self):
        # The first attention projection receives exactly the attention input.
        return self.network['attention_query']

    def forward(self, x, xt):
        return self.network(image=x, context=xt)


def _legacy_feature_name(name):
    """State-digest names of the historical three-stage feature container."""
    if name.startswith('conv1.'):
        return '0.0.' + name[len('conv1.'):]
    if name.startswith('bn1.'):
        return '0.1.' + name[len('bn1.'):]
    for stage, prefix in ((1, '0.4.'), (2, '1.'), (3, '2.')):
        if name.startswith(f'layer{stage}.'):
            return prefix + name[len(f'layer{stage}.'):]
    return None


def _load_resnet_features(path, feature_state_sha256, feature_size, networks):
    state = torch.load(path, map_location='cpu', weights_only=True)
    digest = hashlib.sha256()
    stages = []
    cin, size = 3, feature_size
    for stage, cout in ((1, 64), (2, 128), (3, 256)):
        next_size = size // (4 if stage == 1 else 2)
        stem = f'image_resnet_stage{stage}'
        network = build_network(_source(stem, networks), input_shape=('B', cin, size, size),
                                output_shape=('B', cout, next_size, next_size))
        for node in network.plan.nodes:
            module = network[node.id]
            if not module.state_dict():
                continue
            prefix = node.id.replace('_', '.')
            values = {}
            for key in module.state_dict():
                original = key.replace('norm1.', 'bn1.').replace('norm2.', 'bn2.').replace('shortcut.', 'downsample.')
                checkpoint_key = prefix + '.' + original
                if checkpoint_key not in state:
                    # Published ImageNet V1 predates this BN counter; PyTorch's
                    # own loader supplies zero for the same historical format.
                    if key.endswith('num_batches_tracked'):
                        values[key] = module.state_dict()[key]
                    else:
                        raise ValueError(f'Pretrained configuration requires missing checkpoint key {checkpoint_key}')
                else:
                    values[key] = state[checkpoint_key]
            module.load_state_dict(values, strict=True)
            for key, value in module.state_dict().items():
                original = key.replace('norm1.', 'bn1.').replace('norm2.', 'bn2.').replace('shortcut.', 'downsample.')
                legacy = _legacy_feature_name(prefix + '.' + original)
                digest.update(legacy.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        stages.append(network)
        cin, size = cout, next_size
    if digest.hexdigest() != feature_state_sha256:
        raise ValueError('Pretrained feature state SHA256 does not match the declared recipe')
    return nn.ModuleList(stages), digest.hexdigest()


class _FeatureCritic(nn.Module):
    def __init__(self, weights_path, weights_sha256, width, feature_state_sha256, deterministic_features,
                 image_size=32, feature_size=64, attention_seed=124003, networks=None):
        super().__init__()
        self.deterministic_features = deterministic_features
        self.feature_size = feature_size
        path = _verified_file(weights_path, weights_sha256)
        self.features, feature_digest = _load_resnet_features(path, feature_state_sha256, feature_size, networks)
        self.pixel = _PixelDiscriminator(width, image_size, attention_seed, networks)
        self.project = nn.ModuleList([
            build_network(_source('image_feature_head', networks),
                input_shape={name: ('B', ch, feature_size // divisor, feature_size // divisor)
                             for name in ('candidate', 'condition')},
                output_shape=('B', 1))
            for ch, divisor in ((64, 4), (128, 8), (256, 16))])
        self.combine = build_network(_source('image_critic_score', networks),
            input_shape={name: ('B', 1) for name in ('pixel', 'feature1', 'feature2', 'feature3')},
            output_shape=('B', 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.features.eval().requires_grad_(False)
        self.pretrained_metadata = {'weights': 'ResNet18_Weights.IMAGENET1K_V1',
            'weights_sha256': weights_sha256, 'feature_state_sha256': feature_digest,
            'input_size': feature_size, 'stages': ['layer1', 'layer2', 'layer3']}

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.features.requires_grad_(False)
        return self

    @torch.no_grad()
    def condition_features(self, xt):
        h = (F.interpolate(xt, size=self.feature_size, mode='bilinear', align_corners=False) * .5 + .5 - self.mean) / self.std
        result = []
        for block in self.features:
            h = block(h)
            result.append(h)
        return result

    def forward(self, x, xt, condition_features):
        logits = self.pixel(x, xt)
        h = (F.interpolate(x, size=self.feature_size, mode='bilinear', align_corners=False) * .5 + .5 - self.mean) / self.std
        feature_logits = []
        for i, (block, head) in enumerate(zip(self.features, self.project)):
            h = block(h)
            feature_logits.append(head(candidate=h, condition=condition_features[i]))
        logits = self.combine(pixel=logits, **{f'feature{i + 1}': score
                                             for i, score in enumerate(feature_logits)})
        labels = torch.zeros(len(x), device=x.device, dtype=torch.long)
        return ucd_scores(logits, labels, torch.ones_like(labels), num_classes=1,
                          target='time_class', num_steps=1, validate_args=False)


class CIFARDiscriminator(nn.Module):
    """Pixel/feature critic with immutable pretrained ResNet18 features.

    Defaults preserve the 32px CIFAR architecture. Larger
    images add pixel residual stages, keeping attention at 16px and the scalar
    readout at 4px. ``feature_size`` sets the backbone input resolution; 256
    retains native 64/32/16px maps for 256px colorization. The context is always
    a fixed zero image, never another sample or a grayscale condition.
    """
    def __init__(self, weights_path, weights_sha256=RESNET18_SHA256, width=32,
                 feature_state_sha256=FEATURE_SHA256, attention_seed=124003, deterministic_features=True,
                 image_size=32, feature_size=64, networks=None):
        super().__init__()
        if type(deterministic_features) is not bool:
            raise ValueError('deterministic_features must be a boolean')
        for name, size, minimum in [('image_size', image_size, 32), ('feature_size', feature_size, 64)]:
            if type(size) is not int or size < minimum or size & (size - 1):
                raise ValueError(f'{name} must be a power of two >= {minimum}')
        self.image_size = image_size
        self.critic = _FeatureCritic(weights_path, weights_sha256, width, feature_state_sha256,
                                     deterministic_features, image_size, feature_size, attention_seed, networks)
        self.register_buffer('context', torch.zeros(1, 3, image_size, image_size))
        self._context_features = None

    def requires_grad_(self, requires_grad=True):
        self.critic.requires_grad_(requires_grad)
        return self

    def _apply(self, fn, recurse=True):
        self._context_features = None  # Derived cache follows device/dtype moves.
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._context_features = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f'Discriminator requires x [batch,3,{self.image_size},{self.image_size}]')
        if self._context_features is None:
            self._context_features = self.critic.condition_features(self.context)
        features = [v.expand(len(x), -1, -1, -1) for v in self._context_features]
        return self.critic(x, self.context.expand(len(x), -1, -1, -1), features)


class CIFARRoutingEncoder(nn.Module):
    """HNDL encoder with a detached mixture and straight-through routing."""
    def __init__(self, z_dim=64, width=32, temperature=.125, networks=None):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Routing temperature must be positive and finite')
        self.temperature = temperature
        self.z_dim = z_dim
        self.network = build_network(_source('image_encoder', networks,
            width=width, width2=2 * width, width4=4 * width, z_dim=z_dim),
            input_shape=('B', 3, 32, 32), output_shape=('B', 2 * z_dim))

    @property
    def query(self):
        return self.network['query']

    @property
    def offset(self):
        return self.network['offset']

    def forward(self, x, means, sigma):
        query, offset = self.network(x).split(self.z_dim, 1)
        fixed = means.detach()
        distances = (query.square().sum(1, keepdim=True) + fixed.square().sum(1)[None]
                     - 2 * query @ fixed.T) / query.shape[1]
        ids = distances.argmin(1)
        soft = (-distances / self.temperature).softmax(1)
        proxy = soft @ fixed
        center = fixed[ids] + (proxy - proxy.detach())
        bounded = offset
        return {'latent': center + sigma * bounded, 'ids': ids, 'offset': bounded, 'soft': soft}
