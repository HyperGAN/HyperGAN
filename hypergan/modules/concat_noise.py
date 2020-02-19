import torch.nn as nn
import torch
from torch.distributions import uniform

class ConcatNoise(nn.Module):
    def __init__(self):
        super(ConcatNoise, self).__init__()
        self.z = uniform.Uniform(torch.Tensor([-1.0]),torch.Tensor([1.0]))
    def forward(self, x):
        noise = self.z.sample(x.shape).cuda()
        cat = torch.cat([x, noise.view(*x.shape)], 1)
        return cat
