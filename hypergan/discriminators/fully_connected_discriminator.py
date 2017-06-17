import tensorflow as tf
import hyperchamber as hc
import os
import hypergan
from hypergan.discriminators.common import *

from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from .base_discriminator import BaseDiscriminator

class FullyConnectedDiscriminator(BaseDiscriminator):
    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops
        activation = ops.lookup(config.activation or 'lrelu')
        final_activation = ops.lookup(config.final_activation or 'tanh')

        net = ops.linear(net, 512)
        net = activation(net)
        net = ops.linear(net, 512)
        if final_activation:
            net = final_activation(net)

        return net

