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

        net = ops.reshape(net, [ops.shape(net)[0], -1])

        print("[fully connected discriminator] creating FC layer from ", net)
        net = self.layer_regularizer(net)
        net = activation(net)
        net = ops.linear(net, config.features or ops.shape(net)[-1])
        if final_activation:
            net = self.layer_regularizer(net)
            net = final_activation(net)

        self.sample = net

        return net

