import tensorflow as tf
import hyperchamber as hc
from hypergan.discriminators.common import *
import inspect
import os

from .base_discriminator import BaseDiscriminator

class DCGANDiscriminator(BaseDiscriminator):

    def required(self):
        return []

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops
        activation = ops.lookup(config.activation or 'lrelu')
        improved = config.improved

        net = self.add_noise(net)
        net = ops.conv2d(net, 3, 3, 2, 2, 64)
        if improved:
            net = self.layer_regularizer(net)
        net = activation(net)
        for layer in range(3):
            net = ops.conv2d(net, 3, 3, 2, 2, ops.shape(net)[-1]*2)
            net = self.layer_regularizer(net)
            net = activation(net)
        net = ops.reshape(net, [ops.shape(net)[0],-1])
        net = ops.linear(net, config.final_features or 1)
        if improved:
            net = self.layer_regularizer(net)
            net = activation(net)

        return net

