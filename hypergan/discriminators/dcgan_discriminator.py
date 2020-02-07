import hyperchamber as hc
import torch.nn as nn
from hypergan.discriminators.common import *
import inspect
import os

from .base_discriminator import BaseDiscriminator

class DCGANDiscriminator(BaseDiscriminator):

    def required(self):
        return []

    def create(self):
        self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, 2, 1),
                nn.ReLU()
        )
        self.linear = nn.Linear(4*4*512, 1)

    def forward(self, x):
        net = self.net(x).view(self.gan.batch_size(), -1)
        return self.linear(net).view(self.gan.batch_size(),1)
