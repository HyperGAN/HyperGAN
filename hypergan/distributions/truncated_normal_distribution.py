import hyperchamber as hc
import numpy as np
import torch
from .base_distribution import BaseDistribution

from ..gan_component import ValidationException
from torch.distributions import uniform

TINY=1e-12

class TruncatedNormalDistribution(BaseDistribution):
    def __init__(self, gan, config):
        BaseDistribution.__init__(self, gan, config)
        self.current_channels = config["z"]
        self.current_width = 1
        self.current_height = 1
        self.current_input_size = config["z"]
        batch_size = gan.batch_size()
        self.z = torch.Tensor(batch_size, self.current_input_size).cuda()

    def create(self):
        pass

    def required(self):
        return "".split()

    def validate(self):
        errors = BaseDistribution.validate(self)
        #if(self.config.z is not None and int(self.config.z) % 2 != 0):
        #    errors.append("z must be a multiple of 2 (was %2d)" % self.config.z)
        return errors

    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    def truncated_normal_(self, tensor, mean=0, std=1):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

    def sample(self):
        self.truncated_normal_(self.z)
        return self.z

    def next(self):
        self.instance = self.sample()
        return self.instance
