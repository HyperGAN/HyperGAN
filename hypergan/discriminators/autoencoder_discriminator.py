import tensorflow as tf
import hyperchamber as hc
import os
import hypergan
from hypergan.discriminators.common import *

from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from .base_discriminator import BaseDiscriminator

def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return a-b

class AutoencoderDiscriminator(PyramidDiscriminator):

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops

        generator = ResizeConvGenerator(gan, gan.generator.config)
        generator.ops = ops # share variable allocation to make variables part of the discriminator training step

        hidden = PyramidDiscriminator.build(self, net)
        ops.describe("autoencoder")
        reconstruction = generator.build(hidden)

        error = config.distance(net, reconstruction)

        #error = tf.reshape(error, [ops.shape(error)[0], -1])
        #error = tf.concat([error]+mini, axis=1) TODO minibatch

        return error


