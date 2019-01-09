import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator
from .segment_generator import SegmentGenerator

class MultisegmentGenerator(SegmentGenerator):

    def required(self):
        return []

    def build(self, net, mask=None):
        gan = self.gan
        ops = self.ops
        config = self.config

        if(mask is None):
            mask_config  = dict(config.mask_generator)
            data_layers = 6
            mask_config["channels"]=3+data_layers
            mask_config["layer_filter"]=None
            mask_generator = ResizeConvGenerator(gan, mask_config, name='mask', input=net, reuse=self.ops._reuse)
            self.mask_generator = mask_generator

            mask_single_channel = mask_generator.sample
        else:
            mask_single_channel = mask
            mask_generator = self.mask_generator


 
        self.mask_single_channel = mask_generator.sample
        #self.mask = tf.tile(mask_generator.sample, [1,1,1,3])
        self.mask = mask_generator.sample



        def add_mask(gan, config, net):
            mask = mask_single_channel
            s = gan.ops.shape(net)
            shape = [s[1], s[2]]
            return tf.image.resize_images(mask, shape, 1)

 
        config['layer_filter'] = add_mask

        g1 = ResizeConvGenerator(gan, config, input=net, name='g1', reuse=self.ops._reuse)
        g2 = ResizeConvGenerator(gan, config, input=net, name='g2', reuse=self.ops._reuse)
        g3 = ResizeConvGenerator(gan, config, input=net, name='g3', reuse=self.ops._reuse)

        if not self.ops._reuse:
            self.ops.add_weights(mask_generator.variables())
            self.ops.add_weights(g1.variables())
            self.ops.add_weights(g2.variables())
            self.ops.add_weights(g3.variables())

        mask1 = tf.slice(mask_single_channel, [0,0,0,0], [-1,-1,-1,1])
        mask2 = tf.slice(mask_single_channel, [0,0,0,1], [-1,-1,-1,1])
        mask3 = tf.slice(mask_single_channel, [0,0,0,2], [-1,-1,-1,1])


        mask2 = tf.nn.relu(mask2-mask1)
        mask3 = tf.nn.relu(mask3-mask2-mask1)
        self.mask = tf.concat([mask1,mask2,mask3], axis=3)

        #mask = tf.ones_like(mask1)
        sample = tf.zeros_like(gan.inputs.x)
        for applied_mask in [(mask1, g1), (mask2, g2), (mask3, g3)]:
            g = applied_mask[1].sample
            m = applied_mask[0]

            #sample += (1.0-mask)*m*g
            #mask = mask*(1.0-m)
            sample += m*g

        self.g1 = g1
        self.g2 = g2
        self.g3 = g3

        self.g1x = (g1.sample * mask1) + \
                (1.0-mask1) * gan.inputs.x


        self.g2x = (gan.inputs.x * (1.0-mask2)) + \
                (mask2) * g2.sample


        self.g3x = (gan.inputs.x * (1.0-mask3)) + \
                (mask3) * g3.sample


        self.mask_generator = mask_generator

        return sample
