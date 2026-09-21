"""Ordinary CIFAR components ported from Martyn Garcia's ParticleGAN.

Source: feat/cifar-ae-gan-pretrained-encoder, commit
9e9ce96c96948197e21e1171c8394e3819bb0013. See docs/cifar-recipe.md.
These factories never download weights or data.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from particlegan import ucd_scores

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


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, affine_condition=True):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.affine = affine_condition
        self.cond = nn.Linear(emb, 2 * cout if self.affine else cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.leaky_relu(self.n1(x), .2))
        q = self.cond(F.leaky_relu(e, .2))[:, :, None, None]
        h = self.n2(h)
        if self.affine:
            scale, shift = q.chunk(2, 1)
            h = h * (1 + scale) + shift
        else:
            h = h + q
        h = self.c2(F.leaky_relu(h, .2))
        return (self.skip(x) + h) / math.sqrt(2)


def _source_generator_rng(z_dim, width):
    # The source first constructs a residual generator and replaces it inside a
    # fork_rng scope. Retain those draws to reproduce D/E initialization too.
    nn.Linear(z_dim, 4 * width * 4 * 4)
    nn.Linear(z_dim, 4 * width)
    for a, b in [(4 * width, 4 * width), (4 * width, 2 * width), (2 * width, width)]:
        _ResBlock(a, b, 4 * width)
    nn.Conv2d(width, 3, 3, padding=1)


class SAGANAttention(nn.Module):
    """Unscaled single-head spatial attention with active unit residual."""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, max(1, channels // 8), 1, bias=False)
        self.key = nn.Conv2d(channels, max(1, channels // 8), 1, bias=False)
        self.value = nn.Conv2d(channels, max(1, channels // 2), 1, bias=False)
        self.project = nn.Conv2d(max(1, channels // 2), channels, 1, bias=False)

    def forward(self, x):
        b, _, h, w = x.shape
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(x).flatten(2)
        probabilities = torch.bmm(query, key).softmax(dim=-1)
        value = self.value(x).flatten(2)
        attended = torch.bmm(value, probabilities.transpose(1, 2)).reshape(b, -1, h, w)
        return x + self.project(attended)


class CIFARGenerator(nn.Module):
    """Source deconvolutional G with 16x16 SAGAN attention."""
    def __init__(self, z_dim=64, width=32, attention_seed=124003):
        super().__init__()
        _source_generator_rng(z_dim, width)
        with torch.random.fork_rng(devices=[]):
            self.input = nn.Linear(z_dim, 256 * 4 * 4)
            self.input_norm = nn.GroupNorm(8, 256)
            self.output = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.GroupNorm(8, 128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.GroupNorm(8, 64), nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1), nn.Tanh())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(attention_seed)
            self.attention = SAGANAttention(64)

    def forward(self, z):
        h = self.input_norm(self.input(z).reshape(-1, 256, 4, 4)).relu()
        for index, layer in enumerate(self.output):
            h = layer(h)
            if index == 5:
                h = self.attention(h)
        return h


class _PixelDiscriminator(nn.Module):
    def __init__(self, width, image_size=32):
        super().__init__()
        self.emb_dim = 4 * width
        self.input = nn.Conv2d(6, width, 3, padding=1)
        channels = [width, 2 * width] + [4 * width] * (int(math.log2(image_size)) - 3)
        self.blocks = nn.ModuleList([
            _ResBlock(a, b, self.emb_dim, affine_condition=False)
            for a, b in zip(channels, channels[1:])])
        self.output = nn.Linear(width * 4 * 4 * 4, 1)

    def forward(self, x, xt):
        e = x.new_zeros(len(x), self.emb_dim)
        h = self.input(torch.cat([x, xt], 1))
        for block in self.blocks:
            h = F.avg_pool2d(block(h, e), 2)
            if h.shape[-1] == 16:
                h = self.attention(h)
        return self.output(F.leaky_relu(h, .2).flatten(1))


class _DeterministicAdaptivePool(torch.autograd.Function):
    """Source forward with a nonoverlapping, atomic-free analytical adjoint."""
    @staticmethod
    def forward(ctx, x):
        height, width = x.shape[-2:]
        if height % 4 or width % 4:
            raise ValueError('Deterministic feature pooling requires dimensions divisible by four')
        ctx.factors = height // 4, width // 4
        return F.adaptive_avg_pool2d(x, 4)

    @staticmethod
    def backward(ctx, gradient):
        height, width = ctx.factors
        return (gradient / (height * width)).repeat_interleave(height, -2).repeat_interleave(width, -1)


class _DeterministicPool2d(nn.Module):
    def forward(self, x):
        return _DeterministicAdaptivePool.apply(x)


class _FeatureCritic(nn.Module):
    def __init__(self, weights_path, weights_sha256, width, feature_state_sha256, deterministic_features,
                 image_size=32, feature_size=64):
        super().__init__()
        self.deterministic_features = deterministic_features
        self.feature_size = feature_size
        path = _verified_file(weights_path, weights_sha256)
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError('CIFAR pretrained critic requires the hypergan[cifar] extra') from exc
        net = resnet18(weights=None)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.features = nn.ModuleList([
            nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1),
            net.layer2, net.layer3])
        for module in self.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.pixel = _PixelDiscriminator(width, image_size)
        self.project = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2 * ch, 64, 1), nn.GroupNorm(8, 64), nn.LeakyReLU(.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(.2),
            (_DeterministicPool2d() if deterministic_features else nn.AdaptiveAvgPool2d(4)),
            nn.Flatten(), nn.Linear(64 * 16, 1))
            for ch in (64, 128, 256)])
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.features.eval().requires_grad_(False)
        digest = hashlib.sha256()
        for name, value in self.features.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        if digest.hexdigest() != feature_state_sha256:
            raise ValueError('Pretrained feature state SHA256 does not match the declared recipe')
        self.pretrained_metadata = {'weights': 'ResNet18_Weights.IMAGENET1K_V1',
            'weights_sha256': weights_sha256, 'feature_state_sha256': digest.hexdigest(),
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
            feature_logits.append(head(torch.cat([h, condition_features[i]], 1)))
        logits = (logits + sum(feature_logits) / math.sqrt(3)) / math.sqrt(2)
        labels = torch.zeros(len(x), device=x.device, dtype=torch.long)
        return ucd_scores(logits, labels, torch.ones_like(labels), num_classes=1,
                          target='time_class', num_steps=1, validate_args=False)


class CIFARDiscriminator(nn.Module):
    """Pixel/feature critic with immutable pretrained ResNet18 features.

    Defaults preserve the 32px CIFAR recipe and its checkpoint layout. Larger
    images add pixel residual stages, keeping attention at 16px and the scalar
    readout at 4px. ``feature_size`` sets the backbone input resolution; 256
    retains native 64/32/16px maps for 256px colorization. The context is always
    a fixed zero image, never another sample or a grayscale condition.
    """
    def __init__(self, weights_path, weights_sha256=RESNET18_SHA256, width=32,
                 feature_state_sha256=FEATURE_SHA256, attention_seed=124003, deterministic_features=True,
                 image_size=32, feature_size=64):
        super().__init__()
        if type(deterministic_features) is not bool:
            raise ValueError('deterministic_features must be a boolean')
        for name, size, minimum in [('image_size', image_size, 32), ('feature_size', feature_size, 64)]:
            if type(size) is not int or size < minimum or size & (size - 1):
                raise ValueError(f'{name} must be a power of two >= {minimum}')
        self.image_size = image_size
        self.critic = _FeatureCritic(weights_path, weights_sha256, width, feature_state_sha256,
                                     deterministic_features, image_size, feature_size)
        self.register_buffer('context', torch.zeros(1, 3, image_size, image_size))
        self._context_features = None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(attention_seed)
            SAGANAttention(64)  # Source constructs G attention before D attention.
            self.critic.pixel.attention = SAGANAttention((2 if image_size == 32 else 4) * width)

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
    """Scratch encoder; reconstruction trains E through frozen G, never means."""
    def __init__(self, z_dim=64, width=32, temperature=.125):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Routing temperature must be positive and finite')
        self.temperature = temperature
        layers = []
        for a, b in [(3, width), (width, 2 * width), (2 * width, 4 * width)]:
            layers += [nn.Conv2d(a, b, 4, stride=2, padding=1),
                       nn.GroupNorm(8, b), nn.LeakyReLU(.2)]
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.query = nn.Linear(4 * width * 4 * 4, z_dim)
        self.offset = nn.Linear(4 * width * 4 * 4, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x, means, sigma):
        h = self.features(x)
        query = self.query(h)
        query = F.layer_norm(query, (query.shape[1],))
        fixed = means.detach()
        distances = (query.square().sum(1, keepdim=True) + fixed.square().sum(1)[None]
                     - 2 * query @ fixed.T) / query.shape[1]
        ids = distances.argmin(1)
        soft = (-distances / self.temperature).softmax(1)
        proxy = soft @ fixed
        center = fixed[ids] + (proxy - proxy.detach())
        bounded = 3 * torch.tanh(self.offset(h) / 3)
        return {'latent': center + sigma * bounded, 'ids': ids, 'offset': bounded, 'soft': soft}
