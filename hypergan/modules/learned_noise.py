import torch.nn as nn
import torch

class LearnedNoise(nn.Module):
    def __init__(self, batch_size, c, h, w, mul=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = w
        self.c = c
        self.weight = nn.Parameter(torch.randn(1, self.c, 1, 1) * mul)
        self.noise = torch.Tensor(self.batch_size, 1, self.h, self.w).cuda()

    def forward(self, input):
        self.noise.normal_()
        return self.noise * self.weight
