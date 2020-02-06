import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator

class DCGANGenerator(BaseGenerator):

    def required(self):
        return []

    def build(self, net):
        return self.gan.inputs.samples[0]
