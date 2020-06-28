import hyperchamber as hc
import numpy as np
import torch
from .base_distribution import BaseDistribution

from ..gan_component import ValidationException
from torch.distributions import uniform

TINY=1e-12

class UniformDistribution(BaseDistribution):
    def __init__(self, gan, config):
        BaseDistribution.__init__(self, gan, config)
        self.current_channels = config["z"]
        self.current_input_size = config["z"]
        batch_size = gan.batch_size()
        self.z = torch.Tensor(batch_size, self.current_input_size).cuda()
        if self.config.projections is not None:
            self.current_input_size *= len(self.config.projections)
            self.current_channels *= len(self.config.projections)

    def create(self):
        pass

    def required(self):
        return "".split()

    def validate(self):
        errors = BaseDistribution.validate(self)
        #if(self.config.z is not None and int(self.config.z) % 2 != 0):
        #    errors.append("z must be a multiple of 2 (was %2d)" % self.config.z)
        return errors

    def lookup(self, projection):
        if callable(projection):
            return projection
        if projection == 'identity':
            return identity
        if projection == 'sphere':
            return sphere
        if projection == 'gaussian':
            return gaussian
        if projection == 'periodic':
            return periodic
        return self.lookup_function(projection)

    def sample(self):
        self.z.uniform_(-1.0, 1.0)
        if self.config.projections is None:
            return self.z
        projections = []
        for projection in self.config.projections:
            projections.append(self.lookup(projection)(self.config, self.gan, self.z))
        ps = []
        for p in projections:
            ps.append(self.z)
        return torch.cat(ps, -1)

    def next(self):
        self.instance = self.sample()
        return self.instance

def identity(config, gan, net):
    return net

def round(config, gan, net):
    net = torch.round(net)
    return net

def binary(config, gan, net):
    net = torch.gt(net, 0)
    net = torch.type(net, torch.Float)
    return net

def zero(config, gan, net):
    return torch.zeros_like(net)
