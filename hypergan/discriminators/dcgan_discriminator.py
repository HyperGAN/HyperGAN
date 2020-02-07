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
        self.net = nn.Sequential(nn.Linear(64*64*3, 1))

    def forward(self, x):
        return self.net(x.view(self.gan.batch_size(), -1)).view(self.gan.batch_size(),1)
