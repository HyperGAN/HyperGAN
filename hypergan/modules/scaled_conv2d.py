import math
import torch
import torch.nn as nn

import torch.nn.functional as F

class ScaledConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=True,
            downsample=False,
            lr_mul=1.0,
            blur_kernel=[1, 3, 3, 1],
            ):

        super(ScaledConv2d, self).__init__()
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )

    def forward(self, input):
        batch, in_channel, height, width = input.shape
        weight = self.scale * self.weight
        out = F.conv2d(input, weight, padding=self.padding)

        return out

