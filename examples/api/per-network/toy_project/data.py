"""Item-level datasets: the user writes only how to load item ``i``.

Order, shuffling, seeding, resume position and sharding belong to HyperGAN.
"""
import math

import torch


class RingPoints:
    """2-D points on a noisy ring. Items are tensors, so they become ``batch.real``."""

    def __init__(self, count=2048, radius=1.0, noise=0.05, seed=0):
        generator = torch.Generator().manual_seed(seed)
        angle = torch.rand(count, generator=generator) * 2 * math.pi
        ring = torch.stack([angle.cos(), angle.sin()], dim=1) * radius
        self.points = ring + noise * torch.randn(count, 2, generator=generator)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]


class PairedPoints:
    """Paired items: a condition and the target it maps to. Dict items name batch fields."""

    def __init__(self, count=1024, scale=2.0, offset=0.25, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.condition = torch.randn(count, 2, generator=generator)
        self.real = self.condition * scale + offset

    def __len__(self):
        return len(self.real)

    def __getitem__(self, index):
        return {'condition': self.condition[index], 'real': self.real[index]}
