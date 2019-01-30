import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator
from .resizable_generator import ResizableGenerator

class SegmentGenerator(ResizableGenerator):

    def required(self):
        return ['mask_generator']

    def reuse(self, net, mask=None):
        self.ops.reuse()
        net = self.build(net, mask)
        self.ops.stop_reuse()
        return net

    def build(self, net, mask=None):
        gan = self.gan
        ops = self.ops
        config = self.config

        if(mask is None):
            mask_config  = dict(config.mask_generator)
            data_channels = 6 #todo parameterize
            mask_config["channels"]=1+data_channels
            mask_config["layer_filter"]=None
            mask_generator = ResizeConvGenerator(gan, mask_config, name='mask', input=net, reuse=self.ops._reuse)
            self.mask_generator = mask_generator

            mask_single_channel = mask_generator.sample
        else:
            mask_single_channel = mask


        def add_mask(gan, config, net):
            mask = mask_single_channel
            s = gan.ops.shape(net)
            shape = [s[1], s[2]]
            return tf.image.resize_images(mask, shape, 1)

        config['layer_filter'] = add_mask

        g1 = ResizeConvGenerator(gan, config, input=net, name='g1', reuse=self.ops._reuse)
        g2 = ResizeConvGenerator(gan, config, input=net, name='g2', reuse=self.ops._reuse)

        if not self.ops._reuse:
            self.ops.add_weights(self.mask_generator.variables())
            self.ops.add_weights(g1.variables())
            self.ops.add_weights(g2.variables())

        self.g1 = g1
        self.g2 = g2

        single = tf.slice(mask_generator.sample, [0,0,0,0], [-1,-1,-1,1])
        self.mask = tf.tile(single, [1,1,1,3])
        self.mask_single_channel = mask_single_channel

        sample = (g1.sample * self.mask) + \
                 (1.0-self.mask) * g2.sample
        self.g1x = (g1.sample * self.mask) + \
                   (1.0-self.mask) * gan.inputs.x
        self.g2x = (gan.inputs.x * self.mask) + \
                   (1.0-self.mask) * g2.sample

        pe = self.gan.skip_connections.get_shapes("progressive_enhancement")
        if pe is not None and len(pe) > 0:
            for shape in pe:
                pe = self.gan.skip_connections.get_array("progressive_enhancement", shape=shape)
                g1 = pe[-2]
                g2 = pe[-1]
                print("[generator] segment generator combining progressive enhancement layers: ", len(pe))
                resized_mask = tf.image.resize_images(self.mask, [shape[1], shape[2]], 1)
                combine = resized_mask * g1 + (1 - resized_mask) * g2
                self.gan.skip_connections.clear("progressive_enhancement", shape=shape)
                for pe_layer in pe[0:-2]+[combine]:
                    self.gan.skip_connections.set("progressive_enhancement", pe_layer)

        if gan.config.progressive_growing:
            pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")
            last_layer = sample * self.progressive_growing_mask(len(pe_layers))
            s = ops.shape(last_layer)
            img_dims = [s[1],s[2]]
            self.pe_layers = [tf.image.resize_images(elem, img_dims) for i, elem in enumerate(pe_layers)] + [sample]
            self.debug_pe = [self.progressive_growing_mask(i) for i, elem in enumerate(pe_layers)]
 

        return sample
