import torch.nn as nn

from hypergan.modules.modulated_conv2d import EqualLinear

class EzNorm(nn.Module):
    def __init__(self, style_size, content_size, equal_linear=False, dim=1):
        super(EzNorm, self).__init__()
        if equal_linear:
            self.beta = EqualLinear(style_size, content_size, lr_mul=1e-8)
            #self.gamma = EqualLinear(style_size, content_size, lr_mul=0.01)
        else:
            self.beta = nn.Linear(style_size, content_size)
            #self.gamma = nn.Linear(style_size, content_size)
        self.dim = dim

    def forward(self, content, style, epsilon=1e-5):
        N = content.shape[0]
        D = content.shape[self.dim]
        view = [N, 1, 1, 1]
        view[self.dim] = D

        #return content * self.gamma(style).view(*view) + self.beta(style).view(*view)
        return content + self.beta(style).view(*view)
