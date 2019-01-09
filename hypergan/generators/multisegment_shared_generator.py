import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator
from .segment_generator import SegmentGenerator

def add_mask(gan, config, net):
    mask = gan.generator.mask_single_channel
    s = gan.ops.shape(net)
    shape = [s[1], s[2]]
    return tf.image.resize_images(mask, shape, 1)

class MultisegmentSharedGenerator(SegmentGenerator):

    def required(self):
        return []

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config
        activation = ops.lookup(config.activation or 'lrelu')
        activation = ops.lookup(config.final_activation or 'tanh')

        mask_config  = dict(config.mask_generator or config)
        mask_config["channels"]=config.segments * (config.segments_spacing or 1)
        mask_config["layer_filter"]=None
        mask_generator = ResizeConvGenerator(gan, mask_config)
        mask_generator.ops.describe("mask")
        if self.ops._reuse:
            mask_generator.ops.reuse()
        mask_generator.build(net)

        self.mask_single_channel = mask_generator.sample
        #self.mask = tf.tile(mask_generator.sample, [1,1,1,3])
        #self.mask = mask_generator.sample

        mask = mask_generator.sample/2.0+0.5

        config['layer_filter'] = add_mask
        config['channels'] = config.segments * 3

        g1 = ResizeConvGenerator(gan, config)
        g1.ops.describe("g1")
        if self.ops._reuse:
            g1.ops.reuse()
        g1.build(net)

        self.ops.add_weights(mask_generator.variables())
        self.ops.add_weights(g1.variables())

        def get_mask(mask, i, config):
            print('get mask', i*(config.segments_spacing or 1), mask)
            return tf.slice(mask, [0,0,0,i*(config.segments_spacing or 1)], [-1,-1,-1,1])

        def get_image(image, i):
            return tf.slice(image, [0,0,0,i*3], [-1,-1,-1,3])
        masks = [(get_mask(mask, i, config), get_image(g1.sample, i)) for i in range(ops.shape(mask)[3]//(config.segments_spacing or 1))]

        full_mask = tf.ones_like(masks[0][0])
        sample = tf.zeros_like(gan.inputs.x)
        if config.stacked:
            for applied_mask in masks:
                g = applied_mask[1]
                m = applied_mask[0]

                sample += (1.0-m)*sample+m*g
                full_mask = full_mask*(1.0-m)

        else:
            for applied_mask in masks:
                g = applied_mask[1]
                m = applied_mask[0]

                sample += (1.0-full_mask)*m*g
                full_mask = full_mask*(1.0-m)

        self.sample = sample

        print("OUTPUT", self.sample, g1.sample, mask)

        self.g1 = hc.Config({"sample":masks[0][1]})
        self.g2 = hc.Config({"sample":masks[1][1]})

        self.g1x = (masks[0][1] * masks[0][0]) + \
                (1.0-masks[0][0]) * gan.inputs.x


        self.g2x = (gan.inputs.x * masks[0][0]) + \
                (1.0-masks[0][0]) * masks[1][1] 

        self.mask_generator = mask_generator

        self.mask = tf.slice(mask_generator.sample, [0,0,0,0], [-1,-1,-1,3])
        #self.sample = self.g1x
        self.masks = masks

        return self.sample
