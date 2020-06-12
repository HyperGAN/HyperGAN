import torch.nn as nn
import torch

class LearnedNoise(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            if len(image.shape) == 4:
                batch, _, height, width = image.shape
                noise = image.new_empty(batch, 1, height, width).normal_()
            elif len(image.shape) == 5:
                batch, n, _, height, width = image.shape
                noise = image.new_empty(batch, n, 1, height, width).normal_()

        return image + self.weight * noise

