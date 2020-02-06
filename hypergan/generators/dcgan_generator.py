import tensorflow as tf
import numpy as np
import hyperchamber as hc
import torch.nn as nn

from hypergan.generators.common import *

from .base_generator import BaseGenerator

class DCGANGenerator(BaseGenerator):

    def required(self):
        return []

    def create(self):
        self.net = nn.Sequential(nn.Linear(100, 64*64*3))

    def forward(self, x):
        return self.net(x).view(self.gan.batch_size(),3,64,64)
