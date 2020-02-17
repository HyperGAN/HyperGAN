import torch
import torch.nn as nn

class Variational(nn.Module):
    def __init__(self, channels, filter=1, stride=1, padding=0, activation=nn.LeakyReLU):
        super(Variational, self).__init__()
        self.mu_logit = nn.Conv2d(channels, channels, filter, stride, padding, padding_mode="reflect")
        self.sigma_logit = nn.Conv2d(channels, channels, filter, stride, padding, padding_mode="reflect")

    def forward(self, x):
        sigma = self.sigma_logit(x)
        mu = self.mu_logit(x)
        z = mu + torch.exp(0.5 * sigma) * torch.randn_like(sigma)
        self.sigma = sigma.view(x.shape[0], -1)
        self.mu = mu.view(x.shape[0], -1)
        return z


