import torch.nn as nn
import torch

class LearnedNoise(nn.Module):
    def __init__(self, batch_size, c, h, w, mul=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = w
        self.c = c
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise = torch.Tensor(self.batch_size, 1, self.h, self.w).cuda()

    def forward(self, input):
        self.noise = torch.randn_like(self.noise, device=self.noise.device)
        return input + self.noise * self.weight
