import torch.nn as nn

from hypergan.modules.modulated_conv2d import EqualLinear

class EzNorm(nn.Module):
    def __init__(self, style_size, channels, dims, equal_linear=False, use_conv=True, dim=1):
        super(EzNorm, self).__init__()
        if equal_linear:
            self.beta = EqualLinear(style_size, channels, lr_mul=0.01)
        else:
            self.beta = nn.Linear(style_size, channels)
        if dims == 2:
            self.conv = nn.Conv1d(channels, 1, 1, 1, padding = 0)
        else:
            self.conv = nn.Conv2d(channels, 1, 1, 1, padding = 0)
        self.dim = dim

    def forward(self, content, style, epsilon=1e-5):
        N = content.shape[0]
        D = content.shape[self.dim]
        view = [1 for x in content.shape]
        view[0] = N
        view[self.dim] = D

        return content + self.beta(style).view(*view) * self.conv(content)
