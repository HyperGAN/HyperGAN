import hyperchamber as hc
from hypergan.discriminators.common import *
import inspect
import os

from .base_discriminator import BaseDiscriminator

class DCGANDiscriminator(BaseDiscriminator):

    def required(self):
        return []

    def build(self, net):
        return self.gan.inputs.samples[0]

