import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator
from .resize_conv_generator import ResizeConvGenerator
from .segment_generator import SegmentGenerator

def add_mask(gan, config, net):
    mask = gan.generator.mask_single_channel
    s = gan.ops.shape(net)
    shape = [s[1], s[2]]
    return tf.image.resize_images(mask, shape, 1)

class MultisegmentGenerator(SegmentGenerator):

    def required(self):
        return []

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config
        activation = ops.lookup(config.activation or 'lrelu')
        activation = ops.lookup(config.final_activation or 'tanh')

        mask_config  = dict(config.mask_generator or config)
        mask_config["channels"]=3
        mask_config["layer_filter"]=None
        mask_generator = ResizeConvGenerator(gan, mask_config)
        mask_generator.ops.describe("mask")
        if self.ops._reuse:
            mask_generator.ops.reuse()
        mask_generator.build(net)

        self.mask_single_channel = mask_generator.sample
        #self.mask = tf.tile(mask_generator.sample, [1,1,1,3])
        self.mask = mask_generator.sample

        mask = mask_generator.sample/2.0+0.5

        config['layer_filter'] = add_mask

        g1 = ResizeConvGenerator(gan, config)
        g1.ops.describe("g1")
        if self.ops._reuse:
            g1.ops.reuse()
        g1.build(net)
        g2 = ResizeConvGenerator(gan, config)
        if self.ops._reuse:
            g2.ops.reuse()
        g2.ops.describe("g2")
        g2.build(net)

        g3 = ResizeConvGenerator(gan, config)
        if self.ops._reuse:
            g3.ops.reuse()
        g3.ops.describe("g3")
        g3.build(net)

        g4 = ResizeConvGenerator(gan, config)
        if self.ops._reuse:
            g4.ops.reuse()
        g4.ops.describe("g4")
        g4.build(net)

        self.ops.add_weights(mask_generator.variables())
        self.ops.add_weights(g1.variables())
        self.ops.add_weights(g2.variables())
        self.ops.add_weights(g3.variables())
        self.ops.add_weights(g4.variables())

        mask1 = tf.slice(mask, [0,0,0,0], [-1,-1,-1,1])
        mask2 = tf.slice(mask, [0,0,0,1], [-1,-1,-1,1])
        mask3 = tf.slice(mask, [0,0,0,2], [-1,-1,-1,1])

        mask = tf.ones_like(mask1)
        sample = tf.zeros_like(gan.inputs.x)
        for applied_mask in [(mask1, g1), (mask2, g2), (mask3, g3)]:
            g = applied_mask[1].sample
            m = applied_mask[0]

            sample += (1.0-mask)*m*g
            mask = mask*(1.0-m)

        self.sample = sample

        print("OUTPUT", self.sample, g1.sample, g2.sample, mask)

        self.g1 = g1
        self.g2 = g2

        self.g1x = (g1.sample * mask1) + \
                (1.0-mask1) * gan.inputs.x


        self.g2x = (gan.inputs.x * mask2) + \
                (1.0-mask2) * g2.sample 

        self.mask_generator = mask_generator

        #self.sample = self.g1x

        return self.sample
