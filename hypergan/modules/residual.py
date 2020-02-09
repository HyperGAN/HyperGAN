import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, channels, filter=3, stride=1, padding=1, activation=nn.LeakyReLU):
        super(Residual, self).__init__()
        self.conv = nn.Conv2d(channels, channels, filter, stride, padding)
        self.activation = activation()
    def forward(self, x):
        return x + self.activation(self.conv(x))
