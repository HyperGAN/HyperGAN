"""A user's project code: networks, datasets, a sampler, a metric, an evaluation.

Nothing here imports HyperGAN internals. Model files name these objects by
import path (``toy_project:MLPGenerator``); HyperGAN imports them where it runs.
"""
import math

import torch
from torch import nn


# -- networks ------------------------------------------------------------------

class MLPGenerator(nn.Module):
    """An ordinary nn.Module. Its forward keywords are bound in the model file."""

    def __init__(self, z_dim=4, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


# -- item-level datasets: only "how to load item i" -----------------------------

class GridPoints:
    """Points around a side x side grid of Gaussian centres in [-1, 1]^2.

    ``split`` selects disjoint deterministic item sets, so evaluations can use
    held-out points. HyperGAN owns batching, shuffling, seeding and resume.
    """

    def __init__(self, side=5, noise=0.02, size=2048, split='train'):
        self.side, self.noise, self.size, self.split = side, noise, size, split
        self.offset = {'train': 0, 'holdout': 10_000_000}[split]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(self.offset + index)
        cell = int(torch.randint(self.side * self.side, (1,), generator=g))
        centre = torch.tensor([cell // self.side, cell % self.side], dtype=torch.float32)
        centre = centre / max(1, self.side - 1) * 2 - 1
        return {'real': centre + self.noise * torch.randn(2, generator=g)}

    def identity(self):
        return {'version': 1}


class PairedVectors:
    """Paired synthetic vectors: real = 2 * condition + 0.25."""

    def __init__(self, size=512, dims=2):
        self.size, self.dims = size, dims

    def __len__(self):
        return self.size

    def load(self, index):
        g = torch.Generator().manual_seed(index)
        condition = torch.randn(self.dims, generator=g)
        return {'condition': condition, 'real': condition * 2.0 + 0.25}


# -- a sampler: generator output -> something to look at ------------------------

def scatter(sample, size=64, extent=1.5):
    """Render 2-D points as a white-on-black image (any number of points)."""
    import hypergan.api as hg
    points = sample.data
    canvas = torch.full((size, size), -1.0)
    xy = ((points.clamp(-extent, extent) + extent) / (2 * extent) * (size - 1)).round().long()
    canvas[size - 1 - xy[:, 1], xy[:, 0]] = 1.0
    return hg.image(canvas)


# -- a metric: a function of values computed anyway -----------------------------

def d_over_g(d, g):
    return d / g


# -- an evaluation: expensive, from an EMA snapshot, on held-out data -----------

def modes_covered(generated, reference, side=5, radius=0.25):
    """Fraction of grid centres with at least one generated point within radius."""
    axis = torch.linspace(-1, 1, side)
    centres = torch.cartesian_prod(axis, axis)
    distance = torch.cdist(generated.float(), centres)
    return float((distance.min(dim=0).values < radius).float().mean())


def mean_nearest_distance(generated, reference):
    """Mean distance from each generated point to its nearest held-out point."""
    return float(torch.cdist(generated.float(), reference.float()).min(dim=1).values.mean())


def tone(sample, rate=8000, seconds=0.25):
    """An audio sampler for 1-D outputs: each value becomes a short tone."""
    import hypergan.api as hg
    frequencies = 440.0 * (2.0 ** sample.data.flatten()[:4].clamp(-1, 1))
    t = torch.arange(int(rate * seconds)) / rate
    return hg.audio(torch.cat([0.3 * torch.sin(2 * math.pi * f * t) for f in frequencies]), rate=rate)
