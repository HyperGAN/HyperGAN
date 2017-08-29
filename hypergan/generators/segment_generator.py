import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator
from .resize_conv_generator import ResizeConvGenerator

def add_mask(gan, config, net):
    mask = gan.generator.mask_single_channel
    s = gan.ops.shape(net)
    shape = [s[1], s[2]]
    return tf.image.resize_images(mask, shape, 1)

class SegmentGenerator(ResizeConvGenerator):

    def required(self):
        return []

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config
        activation = ops.lookup(config.activation or 'lrelu')
        activation = ops.lookup(config.final_activation or 'tanh')

        mask_config  = dict(config.mask_generator or config)
        mask_config["channels"]=1
        mask_config["layer_filter"]=None
        mask_generator = ResizeConvGenerator(gan, mask_config)
        mask_generator.ops.describe("mask")
        if self.ops._reuse:
            mask_generator.ops.reuse()
        mask_generator.build(net)

        self.mask_single_channel = mask_generator.sample
        self.mask = tf.tile(mask_generator.sample, [1,1,1,3])

        if config.mask_generator:
            mask = mask_generator.sample
        else:
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

        self.ops.add_weights(mask_generator.variables())
        self.ops.add_weights(g1.variables())
        self.ops.add_weights(g2.variables())

        self.sample = (g1.sample * mask) + \
                      (1.0-mask) * g2.sample 

        print("OUTPUT", self.sample, g1.sample, g2.sample, mask)

        self.g1 = g1
        self.g2 = g2

        self.g1x = (g1.sample * mask) + \
                (1.0-mask) * gan.inputs.x
        self.g2x = (gan.inputs.x * mask) + \
                (1.0-mask) * g2.sample

        self.mask_generator = mask_generator

        return self.sample
