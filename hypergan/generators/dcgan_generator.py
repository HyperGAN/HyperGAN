import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator

class DCGANGenerator(BaseGenerator):

    def required(self):
        return []

    def create(self):
        gan = self.gan
        ops = self.ops
        ops.describe("generator")
        return self.build(gan.encoder.sample)

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config

        net = ops.linear(gan.encoder.sample, 4*4*1024)

        shape = ops.shape(net)

        net = ops.reshape(net, [shape[0],4,4,1024])

        net = ops.deconv2d(net, 5, 5, 2, 2, 512)
        net = ops.lookup('batch_norm')(self, net)
        net = ops.lookup('lrelu')(net)
        net = ops.deconv2d(net, 5, 5, 2, 2, 256)
        net = ops.lookup('batch_norm')(self, net)
        net = ops.lookup('lrelu')(net)
        net = ops.deconv2d(net, 5, 5, 2, 2, 128)
        net = ops.lookup('batch_norm')(self, net)
        net = ops.lookup('lrelu')(net)
        net = ops.deconv2d(net, 5, 5, 2, 2, gan.channels())
        net = ops.lookup('batch_norm')(self, net)
        net = ops.lookup('tanh')(net)

        self.sample = net
        return self.sample
