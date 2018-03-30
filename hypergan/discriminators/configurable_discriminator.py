import tensorflow as tf
import hyperchamber as hc
import inspect
import os

from .base_discriminator import BaseDiscriminator

class ConfigurableDiscriminator(BaseDiscriminator):

    def __init__(self, gan, config, name=None, input=None, reuse=None, x=None, g=None, features=[]):
        self.layer_ops = {
            "conv": self.layer_conv,
            "control": self.layer_controls,
            "linear": self.layer_linear,
            "reshape": self.layer_reshape,
            "conv_dts": self.layer_conv_dts,
            "squash": self.layer_squash,
            "avg_pool": self.layer_avg_pool,
            "image_statistics": self.layer_image_statistics,
            "combine_features": self.layer_combine_features,
            "resnet": self.layer_resnet,
            "activation": self.layer_activation
            }
        self.features = features
        self.controls = {}
        BaseDiscriminator.__init__(self, gan, config, name=name, input=input,reuse=reuse, x=x, g=g)

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

    def layer_resnet(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops
        depth = int(args[0])
        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)
        stride = options.stride or 1
        stride = int(stride)
        shortcut = net
        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            print("Constucting layer",stddev)
            initializer = ops.random_initializer(float(stddev))()

        if config.defaults.avg_pool:
            net = ops.conv2d(net, 3, 3, 1, 1, depth, initializer=initializer)
            if stride != 1:
                ksize = [1,stride,stride,1]
                net = tf.nn.avg_pool(net, ksize=ksize, strides=ksize, padding='SAME')
        else:
            net = ops.conv2d(net, 3, 3, stride, stride, depth, initializer=initializer)
        net = activation(net)
        net = ops.conv2d(net, 3, 3, 1, 1, depth, initializer=initializer)
        if ops.shape(net)[-1] != ops.shape(shortcut)[-1] or stride != 1:
            if config.defaults.avg_pool:
                shortcut = ops.conv2d(shortcut, 3, 3, 1, 1, depth, initializer=initializer)
                if stride != 1:
                    ksize = [1,stride,stride,1]
                    shortcut = tf.nn.avg_pool(shortcut, ksize=ksize, strides=ksize, padding='SAME')
            else:
                shortcut = ops.conv2d(shortcut, 3, 3, stride, stride, depth, initializer=initializer)
        net = shortcut + net
        net = activation(net)

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

    def layer_reshape(self, net, args, options):
        dims = [int(x) for x in args[0].split("*")]
        dims = [self.ops.shape(net)[0]] + dims
        net = tf.reshape(net, dims)
        return net

    def layer_avg_pool(self, net, args, options):

        options = hc.Config(options)
        stride=options.stride or 2
        stride=int(stride)
        ksize = [1,stride,stride,1]
        net = tf.nn.avg_pool(net, ksize=ksize, strides=ksize, padding='SAME')

        return net 

    def layer_combine_features(self, net, args, options):
        op = None
        if len(args) > 1:
            op = args[0]
            layers = int(args[1])
        print("Combining features", self.features, net)

        for feature in self.features:
            if op == "conv":
                options['stride']=[1,1]
                options['avg_pool']=[1,1]
                feature = self.layer_conv(feature, [layers], options)

            print("Combining features", [net, feature])
            net = tf.concat([net, feature], axis=3)

        return net


    def layer_image_statistics(self, net, args, options):
        s = self.ops.shape(net)
        options = hc.Config(options)
        batch_size = s[0]
        s[-1]=3
        s[0]=-1

        mean, variance = tf.nn.moments(net, [1])
        net = tf.concat([mean,variance], axis=1)
        net = tf.reshape(net, [batch_size, -1])

        return net
        

    def layer_activation(self, net, args, options):
        options = hc.Config(options)
        activation_s = options.activation or self.config.defaults.activation
        activation = self.ops.lookup(activation_s)
        return activation(net)
    
    def layer_squash(self, net, args, options):
        s = self.ops.shape(net)
        batch_size = s[0]
        mean, variance = tf.nn.moments(net, [1,2])
        net = tf.concat([mean,variance], axis=1)
        net = tf.reshape(net, [batch_size, -1])

        return net

    def layer_conv_dts(self, net, args, options):
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
        net = tf.depth_to_space(net, 2)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

    def layer_controls(self, net, args, options):
        self.controls[args[0]] = net

        return net
