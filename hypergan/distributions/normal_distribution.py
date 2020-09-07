import hyperchamber as hc
import numpy as np
import torch
from .base_distribution import BaseDistribution

from ..gan_component import ValidationException

TINY=1e-12

class NormalDistribution(BaseDistribution):
    def __init__(self, gan, config):
        BaseDistribution.__init__(self, gan, config)
        self.current_channels = config["z"]
        self.current_input_size = config["z"]
        batch_size = gan.batch_size()
        self.shape = [batch_size, self.current_input_size]
        self.next()

    def create(self):
        pass

    def required(self):
        return "".split()

    def validate(self):
        errors = BaseDistribution.validate(self)
        #if(self.config.z is not None and int(self.config.z) % 2 != 0):
        #    errors.append("z must be a multiple of 2 (was %2d)" % self.config.z)
        return errors

    def sample(self):
        self.z = torch.randn(self.shape, device=self.gan.device)
        return self.z

    def next(self):
        self.instance = self.sample()
        return self.instance
