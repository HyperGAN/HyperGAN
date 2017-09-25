import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator
from .resize_conv_generator import ResizeConvGenerator

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
        mask_generator = ResizeConvGenerator(gan, mask_config, name='mask', input=net, reuse=self.ops._reuse)

        mask_single_channel = mask_generator.sample
        def add_mask(gan, config, net):
            mask = mask_single_channel
            s = gan.ops.shape(net)
            shape = [s[1], s[2]]
            return tf.image.resize_images(mask, shape, 1)


        if config.mask_generator:
            mask = mask_generator.sample
        else:
            mask = mask_generator.sample/2.0+0.5

        config['layer_filter'] = add_mask

        g1 = ResizeConvGenerator(gan, config, input=net, name='g1', reuse=self.ops._reuse)
        g2 = ResizeConvGenerator(gan, config, input=net, name='g2', reuse=self.ops._reuse)

        sample = (g1.sample * mask) + \
                      (1.0-mask) * g2.sample 

        print("OUTPUT", sample, g1.sample, g2.sample, mask)

        if not hasattr(self, 'g1'):
            self.ops.add_weights(mask_generator.variables())
            self.ops.add_weights(g1.variables())
            self.ops.add_weights(g2.variables())


            self.g1 = g1
            self.g2 = g2

            self.g1x = (g1.sample * mask) + \
                    (1.0-mask) * gan.inputs.x
            self.g2x = (gan.inputs.x * mask) + \
                    (1.0-mask) * g2.sample

            self.mask_generator = mask_generator
            self.mask = tf.tile(mask_generator.sample, [1,1,1,3])
            if config.mask_generator:
                self.mask = self.mask * 2 - 1

        else:
            print("-->SKIP")


        return sample
