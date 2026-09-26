"""A user's own project module for the graph API demos.

Everything the config file references lives here and is imported by path
(``my_project:Generator``), so every training process builds the same thing.
Run the demos from this directory, or put it on PYTHONPATH.
"""
import math

import torch
from torch import nn

import hypergan.graph as hg


# Networks -------------------------------------------------------------------

def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden),
                         nn.LeakyReLU(0.2), nn.Linear(hidden, outputs))


class Generator(nn.Module):
    def __init__(self, z_dim=4, out=2, hidden=64):
        super().__init__()
        self.net = mlp(z_dim, out, hidden)

    def forward(self, z):
        return self.net(z)


class Encoder(nn.Module):
    """Maps a condition to a code; returns a dict to show nested outputs."""
    def __init__(self, dim=2, code=4, hidden=32):
        super().__init__()
        self.net = mlp(dim, 2 * code, hidden)
        self.code = code

    def forward(self, x):
        mu, log_scale = self.net(x).split(self.code, dim=1)
        return {"mu": mu, "scale": log_scale.exp()}


class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=4, code=4, out=2, hidden=32):
        super().__init__()
        self.net = mlp(z_dim + code, out, hidden)

    def forward(self, z, code):
        return self.net(torch.cat([z, code], dim=1))


class Decoder(nn.Module):
    """Reconstructs the condition from the code (an autoencoder path)."""
    def __init__(self, code=4, out=2, hidden=32):
        super().__init__()
        self.net = mlp(code, out, hidden)

    def forward(self, code):
        return self.net(code)


class PairCritic(nn.Module):
    def __init__(self, dim=2, hidden=32):
        super().__init__()
        self.net = mlp(2 * dim, 1, hidden)

    def forward(self, x, condition):
        return self.net(torch.cat([x, condition], dim=1))


# Item-level data --------------------------------------------------------------

class RingItems:
    """``count`` points on a noisy ring; item i is ``{"x": tensor[2]}``.

    This is all a user writes for data: length and item i. HyperGAN owns
    shuffling, batching, seeding, resume position and sharding.
    """
    def __init__(self, count=512, radius=1.0, noise=0.02, seed=0):
        generator = torch.Generator().manual_seed(seed)
        angle = torch.rand(count, generator=generator) * 2 * math.pi
        self.points = torch.stack([angle.cos(), angle.sin()], 1) * radius
        self.points += noise * torch.randn(count, 2, generator=generator)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return {"x": self.points[index]}

    def identity(self):
        return {"kind": "ring", "count": len(self.points)}


# Losses, metrics, evaluations and samplers -------------------------------------

def radius_loss(input, radius=1.0):
    """A custom generator-side loss: keep samples near the ring radius."""
    return (input.norm(dim=1) - radius).square().mean()


def g_over_d(g_loss, d_loss):
    """A cheap custom metric from values every update computes anyway."""
    return g_loss / max(d_loss, 1e-8)


def ring_error(generated, reference):
    """An expensive evaluation on holdout data: mean radius error against the holdout ring."""
    radius = reference.norm(dim=1).mean()
    return float((generated.norm(dim=1) - radius).abs().mean())


def scatter(points):
    """A sampler: 2-D generator output -> a scatter plot."""
    return hg.view.points(points)


def points_and_codes(points, code):
    """A sampler returning several views from nested outputs."""
    return {"points": hg.view.points(points), "code": hg.view.tensor(code["mu"][:4])}
