import tensorflow as tf
import hyperchamber as hc
import os
import hypergan
from hypergan.discriminators.common import *

import hypergan.discriminators.minibatch_discriminator as minibatch
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
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

        print("ERROR PRE IS ", net)
        x, g = self.split_batch(net)
        print("xg PRE IS ", x, g)

        hidden = PyramidDiscriminator.build(self, net)
        reconstruction = gan.generator.build(hidden) #reuse?

        rx, rg = self.split_batch(reconstruction)

        self.rx = rx
        self.rg = rg

        error = tf.concat([config.distance(x, rx), config.distance(g,rg)], axis=0)
        print("ERROR IS ", error)
        error = tf.reshape(error, [ops.shape(error)[0], -1])
        #error = tf.concat([error]+mini, axis=1) TODO minibatch

        return error


