import tensorflow as tf
import hyperchamber as hc
import os
import hypergan
from hypergan.discriminators.common import *

from hypergan.discriminators.dcgan_discriminator import DCGANDiscriminator
from hypergan.distributions.uniform_distribution import UniformDistribution
from ..base_discriminator import BaseDiscriminator

class CramerDiscriminator(BaseDiscriminator):

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops

        discriminator = DCGANDiscriminator(gan, config)
        discriminator.ops = ops
        encoder = UniformDistribution(gan, gan.config.encoder)

        # careful, this order matters
        g2 = gan.generator.reuse(encoder.create())
        double = tf.concat([net] + [g2, g2], axis=0)
        original = discriminator.build(double)
        d1 = self.split_batch(original, 4)

        dg = ops.concat([d1[2], d1[3]], axis=0) # xs for baseline

        #dx is a sampling of x twice
        dx = ops.concat([d1[0], d1[0]], axis=0) # xs for baseline

        xinput = ops.concat([d1[0], d1[1]], axis=0)

        #dg  is a sampling of g twice

        # net is [x,g] (stacked)
        error = self.f(xinput, dx, dg)
        return error

    # this is from the paper
    def f(self, net, dx, dg):
        # Note: this is currently not working that well. we might need a second sample of X 

        return tf.norm(net - dg, axis=1) - tf.norm(dx, axis=1)
