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
            "conv_double": self.layer_conv_double,
            "conv_reshape": self.layer_conv_reshape,
            "conv": self.layer_conv,
            "phase_shift": self.layer_phase_shift,
            "linear": self.layer_linear,
            "slice": self.layer_slice
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

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)

        stride = options.stride or config.defaults.stride or [2,2]
        fltr = options.filter or config.defaults.filter or [5,5]
        print("ARGS", args)
        depth = int(args[0])

        if type(stride) != type([]):
            stride = [int(stride), int(stride)]

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            print("Constucting latyer",stddev) 
            initializer = ops.random_initializer(float(stddev))()

        if type(fltr) == type(""):
            fltr=[int(fltr), int(fltr)]

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

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)

        print("ARGS", args)
        dims = [int(x) for x in args[0].split("*")]
        size = reduce(operator.mul, dims, 1)

        if len(ops.shape(net)) > 2:
            net = tf.reshape(net, [ops.shape(net)[0], -1])
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

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)
        #layer_regularizer = options.layer_regularizer or config.defaults.layer_regularizer or 'null'
        #layer_regularizer = self.ops.lookup(layer_regularizer)
        #print("___layer", layer_regularizer)

        stride = options.stride or config.defaults.stride or [1,1]
        fltr = options.filter or config.defaults.filter or [5,5]
        if type(fltr) == type(""):
            fltr=[int(fltr), int(fltr)]
        depth = int(args[0])

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            print("Constucting latyer",stddev) 
            initializer = ops.random_initializer(float(stddev))()

        print("NET", net)
        net = tf.image.resize_images(net, [ops.shape(net)[1]*2, ops.shape(net)[2]*2],1)
        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
        #net = layer_regularizer(net)
        print("POTNET", net)
        self.add_progressive_enhancement(net)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

    def layer_conv(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)

        stride = options.stride or config.defaults.stride or [1,1]
        fltr = options.filter or config.defaults.filter or [3,3]
        if type(fltr) == type(""):
            fltr=[int(fltr), int(fltr)]
        depth = int(args[0])

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            print("Constucting latyer",stddev) 
            initializer = ops.random_initializer(float(stddev))()

        print("NET", net)
        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
        print("POTNET", net)
        self.add_progressive_enhancement(net)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

    def layer_conv_reshape(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)

        stride = options.stride or config.defaults.stride or [1,1]
        fltr = options.filter or config.defaults.filter or [3,3]
        if type(fltr) == type(""):
            fltr=[int(fltr), int(fltr)]
        depth = int(args[0])

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            print("Constucting latyer",stddev) 
            initializer = ops.random_initializer(float(stddev))()

        print("NET", net)
        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth*4, initializer=initializer)
        s = ops.shape(net)
        net = tf.reshape(net, [s[0], s[1]*2, s[2]*2, depth])
        print("POTNET", net)
        self.add_progressive_enhancement(net)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net


    def layer_conv_double(self, net, args, options):
        x1 = self.layer_conv(net, args, options)
        y1 = self.layer_conv(net, args, options)
        x2 = self.layer_conv(net, args, options)
        y2 = self.layer_conv(net, args, options)
        x = tf.concat([x1,x2],axis=1)
        y = tf.concat([y1,y2],axis=1)
        net = tf.concat([x,y],axis=2)
        return net


    def layer_slice(self, net, args, options):
        w = int(args[0])
        h = int(args[1])
        net = tf.slice(net, [0,0,0,0], [-1,h,w,-1])
        return net

    def layer_phase_shift(self, net, args, options):
 
        def _phase_shift(I, r):
            def _squeeze(x):
                single_batch = (int(x.get_shape()[0]) == 1)
                x = tf.squeeze(x)
                if single_batch:
                    x_shape = [1]+x.get_shape().as_list()
                    x = tf.reshape(x, x_shape)
                return x

            # Helper function with main phase shift operation
            bsize, a, b, c = I.get_shape().as_list()
            X = tf.reshape(I, (bsize, a, b, r, r))
            X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
            X = tf.split(axis=1, num_or_size_splits=a, value=X)  # a, [bsize, b, r, r]
            X = tf.concat(axis=2, values=[_squeeze(x) for x in X])  # bsize, b, a*r, r
            X = tf.split(axis=1, num_or_size_splits=b, value=X)  # b, [bsize, a*r, r]
            X = tf.concat(axis=2, values=[_squeeze(x) for x in X])  #
            bsize, a*r, b*r
            return tf.reshape(X, (bsize, a*r, b*r, 1))

        def phase_shift(X, r, color=False):
          # Main OP that you can arbitrarily use in you tensorflow code
          if color:
            Xc = tf.split(axis=3, num_or_size_splits=3, value=X)
            X = tf.concat(axis=3, values=[_phase_shift(x, r) for x in Xc])
          else:
            X = _phase_shift(X, r)
          return X

        return phase_shift(net, int(args[0]), color=True)

