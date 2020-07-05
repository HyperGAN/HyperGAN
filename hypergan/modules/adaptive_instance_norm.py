import torch.nn as nn

from hypergan.modules.modulated_conv2d import EqualLinear

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, style_size, content_size, equal_linear=False):
        super(AdaptiveInstanceNorm, self).__init__()
        if equal_linear:
            self.gamma = EqualLinear(style_size, content_size)
            self.beta = EqualLinear(style_size, content_size)
        else:
            self.gamma = nn.Linear(style_size, content_size)
            self.beta = nn.Linear(style_size, content_size)

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def calc_mean_std1d(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 3)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
        return feat_mean, feat_std


    def forward(self, content, style, epsilon=1e-5):
        style = style.view(content.shape[0], -1)
        gamma = self.gamma(style)
        beta = self.beta(style)
        if len(content.shape) == 4:
            c_mean, c_var = self.calc_mean_std(content, epsilon)
        elif len(content.shape) == 3:
            c_mean, c_var = self.calc_mean_std1d(content, epsilon)

        c_std = (c_var + epsilon).sqrt()
        return (1+gamma.view(c_std.shape)) * ((content - c_mean) / c_std) + beta.view(c_std.shape)
