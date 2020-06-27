import torch.nn as nn
import torch

class LearnedNoise(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1, c, h, w)*0.1)

    def forward(self, input):

        return input + self.weight
