import tensorflow as tf
import hyperchamber as hc
import inspect
import os

from .base_discriminator import BaseDiscriminator

class ConfigurableDiscriminator(BaseDiscriminator):

    def __init__(self, gan, config, name=None, input=None, reuse=None):
        self.layer_ops = {
            "conv": self.layer_conv,
            "linear": self.layer_linear
            }
        BaseDiscriminator.__init__(self, gan, config, name=name, input=input,reuse=reuse)

    def required(self):
        return "layers defaults".split()

    def build(self, net):
        config = self.config

        for layer in config.layers:
            net = self.parse_layer(net, layer)

        return net

    def parse_args(self, strs):
        options = {}
        args = []
        print("STRS", strs)
        for x in strs:
            if '=' in x:
                lhs, rhs = x.split('=')
                options[lhs]=rhs
            else:
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

    def layer_conv(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)

        stride = options.stride or config.defaults.stride or [2,2]
        fltr = options.filter or config.defaults.filter or config.filter or [5,5]
        if type(fltr) == type(""):
            fltr = [int(fltr), int(fltr)]
        if type(stride) == type(""):
            stride = [int(stride), int(stride)]
        print("ARGS", args)
        depth = int(args[0])

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            print("Constucting latyer",stddev) 
            initializer = ops.random_initializer(float(stddev))()

        net = self.layer_filter(net)
        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
        avg_pool = options.avg_pool or config.defaults.avg_pool
        if type(avg_pool) == type(""):
            avg_pool = [int(avg_pool), int(avg_pool)]
        if avg_pool:
            ksize = [1,avg_pool[0], avg_pool[1],1]
            stride = ksize
            net = tf.nn.avg_pool(net, ksize=ksize, strides=stride, padding='SAME')
            print("AVG POOL", net)

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

        size = int(args[0])
        net = ops.reshape(net, [ops.shape(net)[0], -1])
        net = ops.linear(net, size)

        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

