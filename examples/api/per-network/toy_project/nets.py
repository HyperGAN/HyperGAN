"""Plain torch modules. Forward keyword names are what ``inputs=`` binds."""
import torch
from torch import nn


def _mlp(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.LeakyReLU(0.2),
                         nn.Linear(width, width), nn.LeakyReLU(0.2), nn.Linear(width, outputs))


class PointGenerator(nn.Module):
    """Latent -> a 2-D point."""

    def __init__(self, z_dim=4, width=64, out_dim=2):
        super().__init__()
        self.net = _mlp(z_dim, width, out_dim)

    def forward(self, z):
        return self.net(z)


class Encoder(nn.Module):
    """Condition -> a code the generator reads."""

    def __init__(self, dim=2, code=2, width=16):
        super().__init__()
        self.net = _mlp(dim, width, code)

    def forward(self, condition):
        return self.net(condition)


class PairCritic(nn.Module):
    """Scores (sample, condition) pairs."""

    def __init__(self, dim=2, condition=2, width=32):
        super().__init__()
        self.net = _mlp(dim + condition, width, 1)

    def forward(self, x, condition):
        return self.net(torch.cat([x, condition], dim=1))


class FixedFeatures(nn.Module):
    """A fixed random projection; declared frozen, so it never trains."""

    def __init__(self, dim=2, features=4, seed=0):
        super().__init__()
        weight = torch.randn(features, dim, generator=torch.Generator().manual_seed(seed))
        self.projection = nn.Linear(dim, features, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(weight)

    def forward(self, x):
        return torch.tanh(self.projection(x))
