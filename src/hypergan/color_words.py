"""Eight-color particle recipe: one generator, shared MoG codebook, no CLIP.

Routing is the logo AE straight-through rule without layer-norm or the 256px
DINO encoder. Distance is mean squared Euclidean distance to detached means
(sum of squares divided by the latent size). Responsibilities are a temperature
softmax. The center is ``means[ids] + (soft @ fixed - (soft @ fixed).detach())``
and the code is ``center + sigma * 3 * tanh(offset/3)``. Offset maps start at
zero, so a new encoder sits on the selected center. The router samples no noise
and does not own a prior.
"""
import math

import torch
from torch import nn


LABELS = ('red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'black')
# Fixed label order. Channels sit on the cube corners in [-1, 1].
CANONICAL = torch.tensor([
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0],
], dtype=torch.float32)
IMAGE_SIZE = 16


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _temperature(value):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Routing temperature must be positive and finite')
    return float(value)


def _zero_linear(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def route_particles(query, offset, means, sigma, temperature):
    """Straight-through particle code. ``means`` is differentiable; softmax uses a detached copy."""
    if query.ndim != 2 or offset.shape != query.shape:
        raise ValueError('query and offset must both be [batch,z]')
    z_dim = query.shape[1]
    if means.ndim != 2 or means.shape[0] < 2 or means.shape[1] != z_dim:
        raise ValueError(f'means must be [particles>=2,{z_dim}]')
    if not isinstance(sigma, torch.Tensor) or sigma.numel() != 1:
        raise ValueError('sigma must be a scalar tensor')
    if means.device != query.device or sigma.device != query.device or means.dtype != query.dtype or offset.dtype != query.dtype:
        raise ValueError('query, offset, means and sigma must share device and floating dtype')
    fixed = means.detach()
    distance = (query.square().sum(1, keepdim=True) + fixed.square().sum(1)[None]
                - 2 * query @ fixed.T) / z_dim
    log_probs = (-distance / temperature).log_softmax(1)
    ids = log_probs.argmax(1)
    soft = log_probs.exp()
    proxy = soft @ fixed
    center = means[ids] + (proxy - proxy.detach())
    latent = center + sigma * 3 * torch.tanh(offset / 3)
    return latent, ids


class ColorWords:
    """Noisy solid 16×16 RGB colors. No files and no download.

    Draws consume the caller-owned generator in a fixed order: label indices,
    then pixel noise. Canonical channels are ±1, so the clamp that keeps
    samples in [-1, 1] truncates noise that would leave the cube.
    """
    resume_stateless = True

    def __init__(self, noise=0.05):
        if isinstance(noise, bool) or type(noise) not in (int, float) or not math.isfinite(noise) or noise < 0:
            raise ValueError('noise must be a nonnegative finite number')
        self.noise = float(noise)

    def __call__(self, batch_size, *, generator):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        label = torch.randint(len(LABELS), (batch_size, 1), generator=generator)
        base = CANONICAL[label.view(-1)].view(batch_size, 3, 1, 1).expand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        noise = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator) * self.noise
        return {'real': (base + noise).clamp(-1, 1), 'label': label}


def _image_features(width):
    return nn.Sequential(
        nn.Conv2d(3, width, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(width, width, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Flatten(),
    )


class ImageEncoder(nn.Module):
    """16×16 RGB to a shared-codebook latent. Inputs are image, means and sigma."""
    def __init__(self, z_dim=16, width=8, temperature=0.125):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        width = _positive_integer(width, 'width')
        self.temperature = _temperature(temperature)
        self.features = _image_features(width)
        hidden = width * 4 * 4
        self.query = nn.Linear(hidden, self.z_dim)
        self.offset = _zero_linear(nn.Linear(hidden, self.z_dim))

    def forward(self, image, means, sigma):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f'ImageEncoder requires image [batch,3,{IMAGE_SIZE},{IMAGE_SIZE}]')
        hidden = self.features(image)
        latent, ids = route_particles(self.query(hidden), self.offset(hidden), means, sigma, self.temperature)
        return {'latent': latent, 'ids': ids}


class TextEncoder(nn.Module):
    """Learned 8-way label embedding plus the same router. This is not CLIP."""
    def __init__(self, z_dim=16, embed_dim=16, temperature=0.125):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        embed_dim = _positive_integer(embed_dim, 'embed_dim')
        self.temperature = _temperature(temperature)
        self.embed = nn.Embedding(len(LABELS), embed_dim)
        self.query = nn.Linear(embed_dim, self.z_dim)
        self.offset = _zero_linear(nn.Linear(embed_dim, self.z_dim))

    def forward(self, label, means, sigma):
        if label.ndim == 2 and label.shape[1] == 1:
            label = label.reshape(-1)
        if label.ndim != 1 or label.dtype != torch.int64:
            raise ValueError('TextEncoder requires int64 label [batch] or [batch,1]')
        embedding = self.embed(label)
        latent, ids = route_particles(self.query(embedding), self.offset(embedding), means, sigma, self.temperature)
        return {'embedding': embedding, 'latent': latent, 'ids': ids}


class ZEmbedding(nn.Module):
    """Linear map from a prior latent to the text-critic fake. Not a token generator."""
    def __init__(self, z_dim=16, embed_dim=16):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        self.map = nn.Linear(self.z_dim, _positive_integer(embed_dim, 'embed_dim'))

    def forward(self, latent):
        if latent.ndim != 2 or latent.shape[1] != self.z_dim:
            raise ValueError(f'ZEmbedding requires latent [batch,{self.z_dim}]')
        return self.map(latent)


class ColorGenerator(nn.Module):
    """One shared latent-only RGB generator, bounded with tanh."""
    def __init__(self, z_dim=16, width=8):
        super().__init__()
        self.z_dim = _positive_integer(z_dim, 'z_dim')
        width = _positive_integer(width, 'width')
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, width * 4 * 4),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (width, 4, 4)),
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(width, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, latent):
        if latent.ndim != 2 or latent.shape[1] != self.z_dim:
            raise ValueError(f'ColorGenerator requires latent [batch,{self.z_dim}]')
        return self.net(latent)


class ImageCritic(nn.Module):
    """Scalar score of one 16×16 RGB candidate."""
    def __init__(self, width=8):
        super().__init__()
        width = _positive_integer(width, 'width')
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, width, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(width * 4 * 4, 1),
        )

    def forward(self, candidate):
        if candidate.ndim != 4 or tuple(candidate.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f'ImageCritic requires candidate [batch,3,{IMAGE_SIZE},{IMAGE_SIZE}]')
        return self.net(candidate)


class TextCritic(nn.Module):
    """Scalar score of one embedding candidate."""
    def __init__(self, embed_dim=16, width=16):
        super().__init__()
        embed_dim = _positive_integer(embed_dim, 'embed_dim')
        width = _positive_integer(width, 'width')
        self.net = nn.Sequential(
            nn.Linear(embed_dim, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, 1),
        )

    def forward(self, candidate):
        if candidate.ndim != 2 or candidate.shape[1] != self.net[0].in_features:
            raise ValueError(f'TextCritic requires candidate [batch,{self.net[0].in_features}]')
        return self.net(candidate)


class JointCritic(nn.Module):
    """Scalar score of a small image projection concatenated with the embedding."""
    def __init__(self, embed_dim=16, width=8):
        super().__init__()
        embed_dim = _positive_integer(embed_dim, 'embed_dim')
        width = _positive_integer(width, 'width')
        self.image = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.score = nn.Sequential(
            nn.Linear(width + embed_dim, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, 1),
        )

    def forward(self, candidate, embedding):
        if candidate.ndim != 4 or tuple(candidate.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f'JointCritic requires candidate [batch,3,{IMAGE_SIZE},{IMAGE_SIZE}]')
        if embedding.ndim != 2 or embedding.shape[0] != candidate.shape[0]:
            raise ValueError('JointCritic embedding must be [batch,embed_dim]')
        projected = self.image(candidate)
        return self.score(torch.cat((projected, embedding), 1))


def _labels(label):
    if label.ndim == 2 and label.shape[1] == 1:
        label = label.reshape(-1)
    if label.ndim != 1 or label.dtype != torch.int64:
        raise ValueError('metrics require int64 label [batch] or [batch,1]')
    return label


@torch.no_grad()
def color_word_metrics(real, label, generator, image_encoder, text_encoder, joint_critic, means, sigma):
    """Four measurements from one forward. Nothing is written as an image.

    Label recovery is nearest-canonical (squared RGB distance) of the spatial
    mean of ``G(E(text))``. Chance for eight labels is 0.125. Particle
    agreement without the distinct-id count is not a result. The joint swap gap
    is the true-embedding score minus the score with label ``(label + 1) % 8``;
    the swapped label is forced, not shuffled.
    """
    label = _labels(label)
    image = image_encoder(real, means, sigma)
    text = text_encoder(label.view(-1, 1), means, sigma)
    text_image = generator(text['latent'])
    reconstruction = generator(image['latent'])
    if text_image.shape != real.shape or reconstruction.shape != real.shape:
        raise ValueError('metrics require generated RGB in the real shape')
    if image['ids'].shape != label.shape or text['ids'].shape != label.shape:
        raise ValueError('metrics require one particle id per sample')
    mean_color = text_image.mean(dim=(2, 3))
    palette = CANONICAL.to(device=mean_color.device, dtype=mean_color.dtype)
    nearest = (mean_color[:, None, :] - palette[None]).square().sum(-1).argmin(1)
    recovery = (nearest == label.to(nearest.device)).float().mean()
    text_mae = (text_image - real).abs().mean()
    reconstruction_mae = (reconstruction - real).abs().mean()
    agreement = (image['ids'] == text['ids']).float().mean()
    distinct = int(torch.unique(torch.cat((image['ids'], text['ids']))).numel())
    swapped = text_encoder(((label + 1) % len(LABELS)).view(-1, 1), means, sigma)['embedding']
    true_score = joint_critic(real, text['embedding'])
    swapped_score = joint_critic(real, swapped)
    gap = (true_score - swapped_score).mean()
    return {'label_recovery': float(recovery), 'text_image_mae': float(text_mae),
            'reconstruction_mae': float(reconstruction_mae), 'particle_agreement': float(agreement),
            'distinct_particles': distinct, 'joint_swap_gap': float(gap)}
