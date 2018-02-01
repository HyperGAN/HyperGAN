import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

import operator
from functools import reduce

from .base_generator import BaseGenerator

class ConfigurableGenerator(BaseGenerator):
    def __init__(self, gan, config, name=None, input=None, reuse=False):

        self.layer_ops = {
            "deconv": self.layer_deconv,
            "resize_conv": self.layer_resize_conv,
            "linear": self.layer_linear
            }
        BaseGenerator.__init__(self, gan, config, name=name, reuse=reuse,input=input)


    def required(self):
        return "layers defaults".split()

    def build(self, net):
        gan = self.gan
        config = self.config
        ops = self.ops

        for layer in config.layers:
            net = self.parse_layer(net, layer)

        pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")
        s = ops.shape(net)
        img_dims = [s[1],s[2]]
        self.pe_layers = [tf.image.resize_images(elem, img_dims) for i, elem in enumerate(pe_layers)] + [net]
        if gan.config.progressive_growing:
            last_layer = net * self.progressive_growing_mask(len(pe_layers))
            self.debug_pe = [self.progressive_growing_mask(i) for i, elem in enumerate(pe_layers)]
        #    net = tf.add_n(nets + [last_layer])


        return net

    def parse_args(self, strs):
        options = {}
        args = []
        print("STRS", strs)
        for x in strs:
            if '=' in x:
                print("=Found ", strs)
                lhs, rhs = x.split('=')
                options[lhs]=rhs
            else:
                print("Found ", strs)
                args.append(x)
        return args, options

    def parse_layer(self, net, layer):
        config = self.config

        d = layer.split(' ')
        op = d[0]
        print("layer", layer, d)
        args, options = self.parse_args(d[1:])
        
        return self.build_layer(net, op, args, options)

    def build_layer(self, net, op, args, options):
        if self.layer_ops[op]:
            net = self.layer_ops[op](net, args, options)
        else:
            print("ConfigurableGenerator Op not defined", op)

        return net

    def layer_deconv(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or config.defaults.activation or "lrelu"
        activation = self.ops.lookup(activation_s)

        stride = config.stride or [2,2]
        fltr = config.filter or [5,5]
        print("ARGS", args)
        depth = int(args[0])

        initializer = None # default to global
        if options.stddev:
            print("Constucting latyer",options.stddev) 
            initializer = ops.random_initializer(float(options.stddev))()

        net = ops.deconv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
        self.add_progressive_enhancement(net)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net


    def layer_linear(self, net, args, options):
        options = hc.Config(options)
        ops = self.ops
        config = self.config
        fltr = options.filter or config.defaults.filter

        activation_s = options.activation or config.defaults.activation or "lrelu"
        activation = self.ops.lookup(activation_s)

        print("ARGS", args)
        dims = [int(x) for x in args[0].split("*")]
        size = reduce(operator.mul, dims, 1)

        net = ops.linear(net, size)

        if len(dims) > 1:
            net = ops.reshape(net, [ops.shape(net)[0], dims[0], dims[1], dims[2]])

        self.add_progressive_enhancement(net)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

    def layer_resize_conv(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or config.defaults.activation or "lrelu"
        activation = self.ops.lookup(activation_s)

        stride = config.stride or [1,1]
        fltr = config.filter or [5,5]
        depth = int(args[0])

        initializer = None # default to global
        if options.stddev:
            print("Constucting latyer",options.stddev) 
            initializer = ops.random_initializer(float(options.stddev))()

        print("NET", net)
        net = tf.image.resize_images(net, [ops.shape(net)[1]*2, ops.shape(net)[2]*2],1)
        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
        print("POTNET", net)
        self.add_progressive_enhancement(net)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net


