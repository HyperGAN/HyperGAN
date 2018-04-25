import tensorflow as tf
import hyperchamber as hc
import inspect
import copy
import os
import operator
from functools import reduce

from hypergan.ops.tensorflow.extended_ops import bicubic_interp_2d
from .base_discriminator import BaseDiscriminator

class ConfigurableDiscriminator(BaseDiscriminator):

    def __init__(self, gan, config, name=None, input=None, reuse=None, x=None, g=None, features=[], skip_connections=[]):
        self.layers = []
        self.skip_connections = skip_connections
        self.layer_ops = {
            "phase_shift": self.layer_phase_shift,
            "conv": self.layer_conv,
            "control": self.layer_controls,
            "linear": self.layer_linear,
            "subpixel": self.layer_subpixel,
            "unpool": self.layer_unpool,
            "slice": self.layer_slice,
            "concat_noise": self.layer_noise,
            "variational_noise": self.layer_variational_noise,
            "noise": self.layer_noise,
            "pad": self.layer_pad,
            "fractional_avg_pool": self.layer_fractional_avg_pool,
            "bicubic_conv": self.layer_bicubic_conv,
            "conv_double": self.layer_conv_double,
            "conv_reshape": self.layer_conv_reshape,
            "reshape": self.layer_reshape,
            "conv_dts": self.layer_conv_dts,
            "deconv": self.layer_deconv,
            "resize_conv": self.layer_resize_conv,
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

    def build(self, net, replace_controls={}):
        self.replace_controls=replace_controls
        config = self.config

        for layer in config.layers:
            net = self.parse_layer(net, layer)
            self.layers += [net]

        return net

    def parse_args(self, strs):
        options = {}
        args = []
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

        print("Looking for skip connection", self.skip_connections)
        for sk in self.skip_connections:
            if len(ops.shape(sk)) == len(ops.shape(net)) and ops.shape(sk)[1] == ops.shape(net)[1]:
                print("Adding skip connection")

                net = tf.concat([net, sk], axis=3)

        if config.defaults.adaptive_instance_norm and len(self.features) > 0:
            feature = self.features[0]
            feature = self.layer_linear(feature, [128], options)
            opts = copy.deepcopy(dict(options))
            opts['activation']='null'
            size = self.ops.shape(net)[3]
            feature = self.layer_linear(feature, [size*2], opts)
            f1 = tf.reshape(self.ops.slice(feature, [0,0], [-1, size]), [-1, 1, 1, size])
            f2 = tf.reshape(self.ops.slice(feature, [0,size], [-1, size]), [-1, 1, 1, size])
            net = self.adaptive_instance_norm(net, f1,f2)
            print("NETTT", net, f1, f2)


        stride = options.stride or config.defaults.stride or [2,2]
        fltr = options.filter or config.defaults.filter or config.filter or [5,5]
        if type(fltr) == type(""):
            fltr = [int(fltr), int(fltr)]
        if type(stride) == type(""):
            stride = [int(stride), int(stride)]
        depth = int(args[0])

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            initializer = ops.random_initializer(float(stddev))()

        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
        avg_pool = options.avg_pool or config.defaults.avg_pool
        if type(avg_pool) == type(""):
            avg_pool = [int(avg_pool), int(avg_pool)]
        if avg_pool:
            ksize = [1,avg_pool[0], avg_pool[1],1]
            stride = ksize
            net = tf.nn.avg_pool(net, ksize=ksize, strides=stride, padding='SAME')

        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net


    def layer_linear(self, net, args, options):
        print('--net', net)

        options = hc.Config(options)
        ops = self.ops
        config = self.config
        fltr = options.filter or config.defaults.filter

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)


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
        stride=options.stride or self.ops.shape(net)[1]
        stride=int(stride)
        ksize = [1,stride,stride,1]
        net = tf.nn.avg_pool(net, ksize=ksize, strides=ksize, padding='SAME')

        return net 

    def layer_combine_features(self, net, args, options):
        op = None
        print("Combining features", self.features, net)
        if(len(args) > 0):
            op = args[0]

        for feature in self.features:
            if op == "conv":
                options['stride']=[1,1]
                options['avg_pool']=[1,1]
                layers = int(args[1])
                feature = self.layer_conv(feature, [layers], options)

            if op == "linear":
                feature = self.layer_linear(feature, [args[1]], options)
                feature = self.layer_reshape(feature, [args[2]], options)

            if feature is not None:
                print("Combining features", [net, feature])
                net = tf.concat([net, feature], axis=len(self.ops.shape(net))-1)

        return net

    def adaptive_instance_norm(self, content, gamma, beta, epsilon=1e-5):
        c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
        c_std = tf.sqrt(c_var + epsilon)
        return gamma * ((content - c_mean) / c_std) + beta



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
        net = tf.reshape(net, [batch_size, -1])
        net = tf.reduce_mean(net, axis=1, keep_dims=True)

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
            initializer = ops.random_initializer(float(stddev))()

        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth*4, initializer=initializer)
        s = ops.shape(net)
        net = tf.depth_to_space(net, 2)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

    def layer_controls(self, net, args, options):
        if args[0] in self.replace_controls:
            return self.replace_controls[args[0]]
        self.controls[args[0]] = net

        return net

    def layer_resize_conv(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)
        #layer_regularizer = options.layer_regularizer or config.defaults.layer_regularizer or 'null'
        #layer_regularizer = self.ops.lookup(layer_regularizer)

        stride = options.stride or config.defaults.stride or [1,1]
        fltr = options.filter or config.defaults.filter or [5,5]
        if type(fltr) == type(""):
            fltr=[int(fltr), int(fltr)]
        depth = int(args[0])

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            initializer = ops.random_initializer(float(stddev))()

        net = tf.image.resize_images(net, [ops.shape(net)[1]*2, ops.shape(net)[2]*2],1)
        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
        #net = layer_regularizer(net)
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

    def layer_bicubic_conv(self, net, args, options):
        s = self.ops.shape(net)
        net = bicubic_interp_2d(net, [s[1]*2, s[2]*2])
        net = self.layer_conv(net, args, options)
        return net

    def layer_deconv(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or config.defaults.activation
        activation = self.ops.lookup(activation_s)

        stride = options.stride or config.defaults.stride or [2,2]
        fltr = options.filter or config.defaults.filter or [5,5]
        depth = int(args[0])

        if type(stride) != type([]):
            stride = [int(stride), int(stride)]

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            initializer = ops.random_initializer(float(stddev))()

        if type(fltr) == type(""):
            fltr=[int(fltr), int(fltr)]

        net = ops.deconv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth, initializer=initializer)
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
        if type(stride) == type(""):
            stride=[int(stride), int(stride)]
        depth = int(args[0])

        initializer = None # default to global
        stddev = options.stddev or config.defaults.stddev or 0.02
        if stddev:
            initializer = ops.random_initializer(float(stddev))()

        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth*4, initializer=initializer)
        s = ops.shape(net)
        net = tf.reshape(net, [s[0], s[1]*2, s[2]*2, depth])
        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)
        return net

    def layer_unpool(self, net, args, options):

        def unpool_2d(pool, 
                       ind, 
                       stride=[1, 2, 2, 1], 
                       scope='unpool_2d'):
           """Adds a 2D unpooling op.
           https://arxiv.org/abs/1505.04366
         
           Unpooling layer after max_pool_with_argmax.
                Args:
                    pool:        max pooled output tensor
                    ind:         argmax indices
                    stride:      stride is the same as for the pool
                Return:
                    unpool:    unpooling tensor
           """
           with tf.variable_scope(scope):
             input_shape = tf.shape(pool)
             output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]
         
             flat_input_size = tf.reduce_prod(input_shape)
             flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
         
             pool_ = tf.reshape(pool, [flat_input_size])
             batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                               shape=[input_shape[0], 1, 1, 1])
             b = tf.ones_like(ind) * batch_range
             b1 = tf.reshape(b, [flat_input_size, 1])
             ind_ = tf.reshape(ind, [flat_input_size, 1])
             ind_ = tf.concat([b1, ind_], 1)
         
             ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
             ret = tf.reshape(pool_, output_shape)
         
             set_input_shape = pool.get_shape()
             set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
             ret.set_shape(set_output_shape)
             return ret
 
 
        options = hc.Config(options)
        net = unpool_2d(net, tf.ones_like(net, dtype=tf.int64))

        return net 


    def layer_fractional_avg_pool(self, net, args, options):
        options = hc.Config(options)
        net,_,_ = tf.nn.fractional_avg_pool(net, [1.0,0.5,0.5,1.0])

        return net 
    def layer_pad(self, net, args, options):
        options = hc.Config(options)
        s = self.ops.shape(net)
        sizew = s[1]//2
        sizeh = s[2]//2
        net,_,_ = tf.pad(net, [[0,0],[ sizew,sizew],[ sizeh,sizeh],[ 0,0]])

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


    def layer_subpixel(self, net, args, options):
        options = hc.Config(options)
        depth = int(args[0])
        r = options.r or 2
        r = int(r)
        def _PS(X, r, n_out_channel):
                if n_out_channel >= 1:
                    bsize, a, b, c = X.get_shape().as_list()
                    bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
                    Xs=tf.split(X,r,3) #b*h*w*r*r
                    Xr=tf.concat(Xs,2) #b*h*(r*w)*r
                    X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c
                else:
                    print("outchannel < 0")
                return X
        args[0]=depth*(r**2)
        y1 = self.layer_conv(net, args, options)
        ps = _PS(y1, r, depth)
        print("NETs", ps, y1, net)
        return ps

    def layer_slice(self, net, args, options):
        if len(args) == 0:
            w = self.gan.width()
            h = self.gan.height()
        else:
            w = int(args[0])
            h = int(args[1])
        net = tf.slice(net, [0,0,0,0], [-1,h,w,-1])
        return net

    def layer_noise(self, net, args, options):
        net += tf.random_normal(self.ops.shape(net), stddev=0.1)
        return net

    def layer_variational_noise(self, net, args, options):
        net *= tf.random_normal(self.ops.shape(net), mean=1, stddev=0.02)
        return net


    def layer_concat_noise(self, net, args, options):
        noise = tf.random_normal(self.ops.shape(net), stddev=0.1)
        net = tf.concat([net, noise], axis=len(self.ops.shape(net))-1)
        return net




