import torch.nn as nn
import torch

class Const(nn.Module):
    def __init__(self, c, h, w, mul=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, c, h, w) * mul)

    def forward(self, _input):
        return self.weight
