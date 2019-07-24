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
    def __init__(self, gan, config, name=None, input=None, reuse=None, features=[], skip_connections=[]):
        ConfigurableComponent.__init__(self, gan, config, name=name, input=input,features=features,reuse=reuse)
        BaseDiscriminator.__init__(self, gan, config, name=name, input=input,features=features,reuse=reuse)

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
