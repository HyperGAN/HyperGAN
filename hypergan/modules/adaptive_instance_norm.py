import torch.nn as nn

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, style_size, content_size):
        super(AdaptiveInstanceNorm, self).__init__()
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

    def forward(self, content, style, epsilon=1e-5):
        gamma = self.gamma(style)
        beta = self.beta(style)
        c_mean, c_var = self.calc_mean_std(content, epsilon)
        c_std = (c_var + epsilon).sqrt()
        return (1+gamma.view(c_std.shape)) * ((content - c_mean) / c_std) + beta.view(c_std.shape)
