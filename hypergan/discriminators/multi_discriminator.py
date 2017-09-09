import tensorflow as tf
import numpy as np
import hyperchamber as hc

from hypergan.discriminators.base_discriminator import BaseDiscriminator
from hypergan.multi_component import MultiComponent

TINY=1e-8
class MultiDiscriminator(BaseDiscriminator):

    def project(self, i, net, reuse=False):
        filter_w = 4
        filter_h = 4
        ops = self.ops

        if self.config.projection_type == 'scaled':
            width = 32
            height = 32
            h = width*height*3
            batch_size = ops.shape(net)[0]
            net = tf.reshape(net, [batch_size, -1])
            w = self.ops.shape(net)[1]
            scaled = tf.random_uniform([w,h], -1, 1, dtype=self.ops.dtype)
            net = tf.matmul(net, scaled)
            net = tf.reshape(net, [batch_size, height, width, 3])
            return net
        else:
            if i == self.config.discriminator_count-1:
                filter_w = 2
                filter_h = 2

            f = ((i+1)**2)


            sq = np.sqrt(3.0/f)

            w = tf.random_uniform([filter_h, filter_w, self.ops.shape(net)[-1], f],
                                   -sq, sq, dtype=self.ops.dtype)

            return tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')

    """Takes multiple distributions and does an additional approximator"""
    def build(self, net):
        gan = self.gan
        config = self.config

        discs = []
        x, g = self.split_batch(net)
        for i in range(config.discriminator_count):
            projection_g = self.project(i, g)
            projection_x = self.project(i, x, reuse=True)
            disc = config['discriminator_class'](gan, config)
            disc.ops.describe(self.ops.description+"_d_"+str(i))
            if self.ops._reuse:
                disc.ops.reuse()
            d_net = disc.create(g=projection_g, x=projection_x)
            if self.ops._reuse:
                disc.ops.stop_reuse()
            self.ops.add_weights(disc.variables())

            discs.append(disc)

        combine = MultiComponent(combine='concat', components=discs)
        return combine.sample


