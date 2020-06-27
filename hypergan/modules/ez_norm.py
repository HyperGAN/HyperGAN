import torch.nn as nn

from hypergan.modules.modulated_conv2d import EqualLinear

class EzNorm(nn.Module):
    def __init__(self, style_size, content_size, channels, equal_linear=False, use_conv=True, dim=1):
        super(EzNorm, self).__init__()
        if equal_linear:
            #self.beta = EqualLinear(style_size, content_size, lr_mul=1e-8)
            layer = nn.Conv2d(channels, 1, 1, 1, padding = (0, 0))
            #self.gamma = EqualLinear(style_size, content_size, lr_mul=0.01)
        else:
            self.beta = nn.Linear(style_size, content_size)
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(channels, 1, 1, 1, padding = (0, 0))
            #self.conv2 = nn.Conv2d(channels, 1, 1, 1, padding = (0, 0))
        self.dim = dim

    def forward(self, content, style, epsilon=1e-5):
        N = content.shape[0]
        D = content.shape[self.dim]
        view = [N, 1, 1, 1]
        view[self.dim] = D

        if self.use_conv:
            return content + self.beta(style).view(*view) * self.conv(content)
        else:
            return content + self.beta(style).view(*view)
