import tensorflow as tf
import hyperchamber as hc
import inspect
import copy
import os
import operator
from functools import reduce

from hypergan.ops.tensorflow.extended_ops import bicubic_interp_2d
from .base_discriminator import BaseDiscriminator
from hypergan.configurable_component import ConfigurableComponent

class ConfigurableDiscriminator(BaseDiscriminator, ConfigurableComponent):
    def __init__(self, gan, config, *args, **kw_args):
        ConfigurableComponent.__init__(self, gan, config,*args, **kw_args)
        self.layer_ops["slice_input"] = self.layer_slice_input
        self.layer_ops["reverse_batch"] = self.layer_reverse_batch
        BaseDiscriminator.__init__(self, gan, config, *args, **kw_args)

    def layer_filter(self, net, args=[], options={}):
        config = self.config
        gan = self.gan
        ops = self.ops
        concats = []

        if 'layer_filter' in config and config.layer_filter is not None:
            print("[discriminator] applying layer filter", config['layer_filter'])
            filters = []
            stacks = self.ops.shape(net)[0] // gan.batch_size()
            for stack in range(stacks):
                piece = tf.slice(net, [stack * gan.batch_size(), 0,0,0], [gan.batch_size(), -1, -1, -1])
                filters.append(ConfigurableComponent.layer_filter(self, piece, args, options))
            layer = tf.concat(axis=0, values=filters)
            concats.append(layer)

        if len(concats) > 1:
            net = tf.concat(axis=3, values=concats)

        return net

    def layer_reverse_batch(self, net, args=[], options={}):
        config = self.config
        gan = self.gan
        ops = self.ops

        size = ops.shape(net)[0]//2
        netx = tf.slice(net, [0,0,0,0], [size,-1,-1,-1])
        netg = tf.slice(net, [size,0,0,0], [size,-1,-1,-1])

        netx_a = tf.split(netx, self.gan.batch_size(), axis=0)
        netg_a = tf.split(netg, self.gan.batch_size(), axis=0)

        netx_a.reverse()
        netg_a.reverse()

        netx_b = tf.concat(netx_a, axis=0)
        netg_b = tf.concat(netg_a, axis=0)
        return tf.concat([netx_b, netg_b], axis=0)

    def layer_slice_input(self, net, args=[], options={}):
        config = self.config
        gan = self.gan
        ops = self.ops

        idx = int(args[0])
        total = int(args[1])

        if len(self.ops.shape(net)) == 4:
            return tf.strided_slice(net, [idx,0,0,0], self.ops.shape(net), [total,1,1,1])
        return tf.strided_slice(net, [idx,0], self.ops.shape(net), [total,1])
