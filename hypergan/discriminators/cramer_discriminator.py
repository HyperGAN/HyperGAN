import tensorflow as tf
import hyperchamber as hc
import os
import hypergan
from hypergan.discriminators.common import *

from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from hypergan.encoders.uniform_encoder import UniformEncoder
from .base_discriminator import BaseDiscriminator

#TODO remove dup
def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return a-b

class CramerDiscriminator(PyramidDiscriminator):

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops

        discriminator = PyramidDiscriminator(gan, config)
        discriminator.ops = ops
        encoder = UniformEncoder(gan, gan.config.encoder)

        # careful, this order matters
        gan.generator.reuse(encoder.create())
        g2 = gan.generator.sample
        d1 = discriminator.build(tf.concat([net, g2], axis=0)) #TODO x2?
        d1 = self.split_batch(d1, 3)

        #dx is a sampling of x twice
        dx = ops.concat([d1[0], d1[0]], axis=0) # xs for baseline
        dg = ops.concat([d1[2], d1[2]], axis=0) # xs for baseline
        original = ops.concat([d1[0], d1[1]], axis=0) # xs for baseline

        #dg  is a sampling of g twice

        # net is [x,g] (stacked)
        print("NET", original)
        print("DX", dx)
        print("DG", dg)
        error = self.f(original, dx, dg)
        return error

    # this is from the paper
    def f(self, net, dx, dg):
        # TODO  TODO might need a second sample of X 
        config = self.config
        gan = self.gan
        ops = self.ops
        distance = config.distance or l2_distance

        return distance(net, dg) - distance(dx, 0)
