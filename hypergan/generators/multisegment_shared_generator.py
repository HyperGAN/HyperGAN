import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator
from .resize_conv_generator import ResizeConvGenerator
from .segment_generator import SegmentGenerator


class MultisegmentSharedGenerator(SegmentGenerator):

    def required(self):
        return []

    def build(self, net, mask=None):
        gan = self.gan
        ops = self.ops
        config = self.config
        activation = ops.lookup(config.activation or 'lrelu')
        activation = ops.lookup(config.final_activation or 'tanh')

        segments = config.segments or 3
        layer_count = segments + segments*3

        if(mask is None):
            mask_config  = dict(config.mask_generator)
            mask_config["channels"]=layer_count
            mask_config["layer_filter"]=None
            mask_generator = ResizeConvGenerator(gan, mask_config, name='mask', input=net, reuse=self.ops._reuse)
            self.mask_generator = mask_generator

            mask_single_channel = mask_generator.sample
        else:
            mask_single_channel = mask
            mask_generator = self.mask_generator

        self.mask_single_channel = mask_single_channel

        def add_mask(gan, config, net):
            s = gan.ops.shape(net)
            shape = [s[1], s[2]]
            return tf.image.resize_images(mask_single_channel, shape, 1)

        config['layer_filter'] = add_mask

        self.ops.add_weights(mask_generator.variables())

        def get_mask(mask, i, config):
            print('get mask', i, mask)
            return tf.slice(mask, [0,0,0,i], [-1,-1,-1,1])

        def get_image(image, i, config):
            g1 = ResizeConvGenerator(gan, config, input=mask_single_channel, name='g'+str(i), reuse=self.ops._reuse)
            return g1.sample

        masks = [[get_mask(mask_single_channel, i, config), get_image(mask_single_channel, i, config)] for i in range(segments)]

        total_mask = tf.zeros_like(masks[0][0])
        sample = tf.zeros_like(masks[0][1])
        for i in range(len(masks)):
            m = masks[i][0]
            g = masks[i][1]
            m = tf.nn.relu(m - total_mask)
            sample += g*m
            total_mask += m
            masks[i][1] = m #TODO mutation ok here?'  maybe separate out these steps

        self.g1 = hc.Config({"sample":masks[0][1]})
        self.g2 = hc.Config({"sample":masks[1][1]})
        self.g3 = hc.Config({"sample":masks[2][1]})

        self.g1x = (masks[0][1] * masks[0][0]) + \
                (1.0-masks[0][0]) * gan.inputs.x

        self.g2x = (masks[1][1] * masks[1][0]) + \
                (1.0-masks[1][0]) * gan.inputs.x

        self.g3x = (masks[2][1] * masks[2][0]) + \
                (1.0-masks[2][0]) * gan.inputs.x

        self.mask_generator = mask_generator

        self.mask = tf.concat([masks[0][1], masks[1][1], masks[2][1]], axis=3)
        self.masks = masks

        return sample
