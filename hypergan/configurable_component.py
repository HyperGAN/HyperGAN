import hyperchamber as hc
import inspect
import copy
import re
import os
import operator
from functools import reduce
import torch.nn as nn

from .gan_component import GANComponent
from hypergan.gan_component import ValidationException

class ConfigurationException(Exception):
    pass

class ConfigurableComponent(GANComponent):
    def __init__(self, gan, config, name=None, input=None, reuse=None, weights=None, biases=None, x=None, g=None, features=[], skip_connections=[], context={}):
        self.current_channels = gan.channels()
        self.current_width = gan.width()
        self.current_height = gan.height()
        self.layers = []
        self.nn_layers = []
        self.skip_connections = skip_connections
        self.layer_options = {}
        self.layer_ops = {
            "activation": self.layer_activation,
            "adaptive_instance_norm": self.layer_adaptive_instance_norm,
            "add": self.layer_add,
            "attention": self.layer_attention,
            "avg_pool": self.layer_avg_pool,
            "bicubic_conv": self.layer_bicubic_conv,
            "combine_features": self.layer_combine_features,
            "concat": self.layer_concat,
            "const": self.layer_const,
            "const_like": self.layer_const_like,
            "control": self.layer_controls,
            "conv": self.layer_conv,
            "conv_depth_to_space": self.layer_conv_dts,
            "conv_double": self.layer_conv_double,
            "conv_reshape": self.layer_conv_reshape,
            "crop": self.layer_crop,
            "deconv": self.layer_deconv,
            "dropout": self.layer_dropout,
            "double_resolution": self.layer_double_resolution,
            "fractional_avg_pool": self.layer_fractional_avg_pool,
            "gram_matrix": self.layer_gram_matrix,
            "identity": self.layer_identity,
            "image_statistics": self.layer_image_statistics,
            "knowledge_base": self.layer_knowledge_base,
            "layer_filter": self.layer_filter,
            "layer_norm": self.layer_layer_norm,
            "layer": self.layer_layer,
            "latent": self.layer_latent,
            "linear": self.layer_linear,
            "match_support": self.layer_match_support,
            "mask": self.layer_mask,
            "minibatch": self.layer_minibatch,
            "noise": self.layer_noise,
            "pad": self.layer_pad,
            "phase_shift": self.layer_phase_shift,
            "pixel_norm": self.layer_pixel_norm,
            "progressive_replace": self.layer_progressive_replace,
            "reduce_sum": self.layer_reduce_sum,
            "reference": self.layer_reference,
            "relational": self.layer_relational,
            "reshape": self.layer_reshape,
            "resize_conv": self.layer_resize_conv,
            "resize_images": self.layer_resize_images,
            "residual": self.layer_residual,
            "slice": self.layer_slice,
            "split": self.layer_split,
            "squash": self.layer_squash,
            "subpixel": self.layer_subpixel,
            "tensorflowcv": self.layer_tensorflowcv,
            "turing_test": self.layer_turing_test,
            "two_sample_stack": self.layer_two_sample_stack,
            "unpool": self.layer_unpool,
            "variational": self.layer_variational,
            "variational_noise": self.layer_variational_noise,
            "zeros": self.layer_zeros,
            "zeros_like": self.layer_zeros_like
            }
        self.features = features
        self.controls = {}
        self.named_layers = {}
        self.context = context
        if not hasattr(gan, "named_layers"):
            gan.named_layers = {}
        self.subnets = hc.Config(hc.Config(config).subnets or {})
        GANComponent.__init__(self, gan, config)

    def required(self):
        return "layers defaults".split()

    def layer(self, name):
        if name in self.gan.named_layers:
            return self.gan.named_layers[name] 
        if name in self.named_layers:
            return self.named_layers[name]
        return None

    def build(self, net, replace_controls={}, context={}):
        self.replace_controls=replace_controls
        config = self.config

        for name, layer in self.context.items():
            self.set_layer(name, layer)

        for name, layer in context.items():
            self.set_layer(name, layer)

        for layer in config.layers:
            net = self.parse_layer(net, layer)
            self.layers += [net]

        return net

    def create(self):
        for layer in self.config.layers:
            net = self.parse_layer(layer)
            self.layers += [net]

        self.net = nn.Sequential(*self.nn_layers)

    def parse_args(self, strs):
        options = {}
        args = []
        for x in strs:
            if '=' in x:
                lhs, rhs = x.split('=', 1)
                options[lhs]=self.gan.configurable_param(rhs)
            else:
                args.append(self.gan.configurable_param(x))
        return args, options

    def parse_layer(self, layer):
        config = self.config

        if isinstance(layer, list):
            ns = []
            axis = -1
            for l in layer:
                if isinstance(l, int):
                    axis = l
                    continue
                n = self.parse_layer(l)
                ns += [n]

        else:
            parens = re.findall('\(.*?\)',layer)
            for i, paren in enumerate(parens):
                layer = layer.replace(paren, "PAREN"+str(i))
            d = layer.split(' ')
            for i, _d in enumerate(d):
                for j, paren in enumerate(parens):
                    d[i] = d[i].replace("PAREN"+str(j), paren)
            op = d[0]
            args, options = self.parse_args(d[1:])
            self.build_layer(op, args, options)

    def build_layer(self, op, args, options):
        if self.layer_ops[op]:
            #before_count = self.count_number_trainable_params()
            self.layer_ops[op](None, args, options)
            if 'name' in options:
                self.set_layer(options['name'], self.nn_layers[-1])

            #after = self.variables()
            #new = set(after) - set(before)
            #for j in new:
            #    self.layer_options[j]=options
            #after_count = self.count_number_trainable_params()
            #if not self.ops._reuse:
            #    if net == None:
            #        print("[Error] Layer resulted in null return value: ", op, args, options)
            #        raise ValidationException("Configurable layer is null")
            #    print("layer: ", self.ops.shape(net), op, args, after_count-before_count, "params")
        else:
            print("ConfigurableComponent: Op not defined", op)

    def set_layer(self, name, net):
        self.gan.named_layers[name] = net
        self.named_layers[name]     = net

    def count_number_trainable_params(self):
        '''
        Counts the number of trainable variables.
        '''
        def get_nb_params_shape(shape):
            '''
            Computes the total number of params for a given shap.
            Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
            '''
            nb_params = 1
            for dim in shape:
                nb_params = nb_params*int(dim)
            return nb_params

        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def do_ops_layer(self, op, net, channels, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        self.ops.activation_name = options.activation_name
        self.ops.activation_trainable = options.trainable

        activation_s = options.activation or self.ops.config_option("activation")
        activation = self.ops.lookup(activation_s)
        initializer = None # default to global

        stride, fltr, avg_pool = self.get_conv_options(config, options)

        trainable = True
        if options.trainable == 'false':
            trainable = False
        if options.initializer is not None:
            initializer = self.ops.lookup_initializer(options.initializer, options)

        name = None
        if options.name:
            name = self.ops.description+options.name
        bias = True
        if options.bias is not None and options.bias.lower() == 'false':
            bias=False


        for sk in self.skip_connections:
            if len(ops.shape(sk)) == len(ops.shape(net)) and ops.shape(sk)[1] == ops.shape(net)[1]:

                net = tf.concat([net, sk], axis=3)

        if op == ops.conv2d:
            net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], channels, initializer=initializer, name=name, trainable=trainable, bias=bias)

        elif op == ops.deconv2d:
            net = ops.deconv2d(net, fltr[0], fltr[1], stride[0], stride[1], channels, name=name, trainable=trainable, bias=bias)

        elif op == ops.linear:
            net = ops.linear(net, channels, initializer=initializer, name=name, trainable=trainable, bias=bias)

        else:
            raise ValidationException("Unknown operation" + op)

        if options.reshape is not None:
            net = tf.reshape(net, [self.ops.shape(net)[0]] + options.reshape)

        if op != ops.linear and (avg_pool[0] > 1 or avg_pool[1] > 1):
            ksize = [1,avg_pool[0], avg_pool[1],1]
            stride = ksize
            net = tf.nn.avg_pool(net, ksize=ksize, strides=stride, padding='SAME')

        if options.adaptive_instance_norm is not None and len(ops.shape(net)) == 4:
            w = options.adaptive_instance_norm
            if w == True:
                w = "w"
            net = self.layer_adaptive_instance_norm(net, [channels], {'w': w, "activation": "null", "initializer": options.adaptive_instance_norm_initializer})

        if options.adaptive_instance_norm2 is not None and len(ops.shape(net)) == 4:
            net = self.layer_adaptive_instance_norm2(net, [channels], {"w": options.adaptive_instance_norm2, "activation": "null", "initializer": options.adaptive_instance_norm_initializer})

        if activation:
            #net = self.layer_regularizer(net)
            net = activation(net)

        self.ops.activation_name = None
        self.ops.activation_trainable = None

        return net


    def layer_filter(self, net, args=[], options={}):
        """
            If a layer filter is defined, apply it.  Layer filters allow for adding information
            to every layer of the network.
        """
        ops = self.ops
        gan = self.gan
        config = self.config
        if config.layer_filter is None:
            return net
        fltr = config.layer_filter(gan, self.config, net)
        if "only" in options:
            return fltr
        if fltr is not None:
            net = ops.concat(axis=3, values=[net, fltr])
        return net


    def layer_residual(self, net, args, options):
        if args == []:
            args = [self.ops.shape(net)[-1]]
        options["stride"] = 1
        options["avg_pool"] = 1
        res = self.layer_conv(net, args, options)

        return net + res
    
    def layer_zeros(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        self.ops.activation_name = options.activation_name
        reshape = [ops.shape(net)[0]] + [int(x) for x in args[0].split("*")]
        size = reduce(operator.mul, reshape)
        net = tf.zeros(reshape)

        return net

    def layer_match_support(self, net, args, options):
        s = self.ops.shape(net)
        s[0] //= 2
        with tf.variable_scope(self.ops.generate_name(), reuse=self.ops._reuse):
            xpx = self.ops.get_weight(s, name='xconst')
        with tf.variable_scope(self.ops.generate_name(), reuse=self.ops._reuse):
            xpg = self.ops.get_weight(s, name='gconst')
        x,g = tf.split(net, 2, axis=0)
        self.named_layers[options['name']+"_mx"] = xpx
        self.named_layers[options['name']+"_mg"] = xpg
        self.named_layers[options['name']+"_m+x"] = x+xpx
        self.named_layers[options['name']+"_m+g"] = g+xpg
        result = tf.concat([x+xpx, g+xpg], axis=0)
        return result

    def layer_mask(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        layer = options.layer
        mask_layer = options.mask_layer

        opts = copy.deepcopy(dict(options))
        opts["name"] = self.ops.description + "/" + (options.mask_name or "mask")
        opts["activation"] = "sigmoid"

        if options.upscale == "subpixel":
            mask = self.layer_subpixel(self.gan.named_layers[options.layer], [1], opts)
        else:
            mask = self.layer_deconv(self.gan.named_layers[options.layer], [1], opts)
        self.set_layer(opts['name'], mask)
        extra = self.gan.named_layers[mask_layer]
        mask = tf.tile(mask, [1,1,1,self.ops.shape(net)[-1]])
        net = (mask) * net + (1-mask) * extra

        return net

    def layer_identity(self, net, args, options):
        if len(args) > 0:
            self.set_layer(args[0], net)
        return net

    def layer_conv(self, net, args, options):
        if len(args) > 0:
            channels = int(args[0])
        else:
            channels = self.ops.shape(net)[-1]

        options = hc.Config(options)

        self.nn_layers.append(nn.Conv2d(self.current_channels, channels, options.filter or 3, options.stride or 2, (options.filter or 3)//2))
        self.nn_layers.append(nn.ReLU())#TODO
        self.current_channels = channels
        self.current_width = self.current_width // 2 #TODO
        self.current_height = self.current_height // 2 #TODO
        self.current_input_size = self.current_channels * self.current_width * self.current_height

    def layer_linear(self, net, args, options):

        self.nn_layers.append(nn.Flatten())#TODO only if necessary
        self.nn_layers.append(nn.Linear(self.current_input_size, int(args[0])))

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

        if options.slice:
            size = [int(x) for x in options.slice.replace("batch_size",str(self.gan.batch_size())).split("*")]
            net = tf.slice(net, [0,0,0,0], size)
        net = tf.nn.avg_pool(net, ksize=ksize, strides=ksize, padding='SAME')

        return net 

    def layer_combine_features(self, net, args, options):
        op = None
        if(len(args) > 0):
            op = args[0]

        def _combine_feature(net, feature, op=None):
            if op == "conv":
                options['stride']=[1,1]
                options['avg_pool']=[1,1]
                layers = int(args[1])
                feature = self.layer_conv(feature, [layers], options)

            if op == "linear":
                feature = self.layer_linear(feature, [args[1]], options)
                feature = self.layer_reshape(feature, [args[2]], options)

            if op == 'gru':
                tanh = tf.tanh
                #tanh = self.ops.prelu()
               # tanh = self.ops.double_sided(default_activation=tanh)
                sigmoid = tf.sigmoid
               # sigmoid = self.ops.double_sided(default_activation=sigmoid)
                def _conv(_net,name, scale=1):
                    _options = dict(options)
                    _options['activation']=None
                    _options['name']=self.ops.description+name
                    return self.layer_conv(_net, [int(args[1])//scale], _options)
                z = sigmoid(_conv(net,'z',scale=2))
                r = tf.sigmoid(_conv(net,'r',scale=2))
                th = _conv(net,'net',scale=2)
                fh = _conv(feature,'feature',scale=2)
                h = tanh(th + fh  * r)
                net = tf.multiply( (1-z), h) + tf.multiply(feature, z)

            if 'only' in options:
                return net
            if feature is not None:
                net = tf.concat([net, feature], axis=len(self.ops.shape(net))-1)
            return net

        if 'name' in options and type(self.features) == type({}):
            return _combine_feature(net, self.features[options['name']], op)

        for feature in self.features:
            net = _combine_feature(net, feature, op)

        return net

    def adaptive_instance_norm(self, content, gamma, beta, epsilon=1e-5):
        c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
        c_std = tf.sqrt(c_var + epsilon)
        return (1+gamma) * ((content - c_mean) / c_std) + beta

    # https://github.com/ftokarev/tf-adain/blob/master/adain/norm.py
    def adaptive_instance_norm2(self, content, style, epsilon=1e-5):
        axes = [1, 2]

        c_mean, c_var = tf.nn.moments(content, axes=axes, keep_dims=True)
        s_mean, s_var = tf.nn.moments(style, axes=axes, keep_dims=True)
        c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

        return s_std * (content - c_mean) / c_std + s_mean

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
        activation_s = options.activation or self.ops.config_option("activation")
        if len(args) > 0:
            activation_s = args[0]
        activation = self.ops.lookup(activation_s)
        return activation(net)

    def layer_turing_test(self, net, args, options):
        """https://arxiv.org/pdf/1810.10948"""
        options = hc.Config(options)
        net2 = tf.reverse(net, [0])
        net = tf.concat([net, net2], axis=3)
        return net

    def layer_dropout(self, net, args, options):
        options = hc.Config(options)
        if self.gan.method != "train":
            return net
        net = tf.nn.dropout(net, (float)(args[0]))
        return net
    
    def layer_split(self, net, args, options):
        options = hc.Config(options)
        axis = len(self.ops.shape(net))-1
        num_splits = int(args[0])
        selected = int(options.select)
        return tf.split(net, num_splits, axis)[selected]

    def layer_slice(self, net, args, options):
        start = int(args[0])
        size = int(args[1])
        bs = self.gan.batch_size()
        if self.ops.shape(net)[1] < start + size:
            raise ValidationException("start " + str(start) + " + size " + str(size) + " of slice is larger than input: "+str(self.ops.shape(net)[1]))
        return tf.slice(net, [0, start], [bs, size])


    def layer_squash(self, net, args, options):
        s = self.ops.shape(net)
        batch_size = s[0]
        net = tf.reshape(net, [batch_size, -1])
        net = tf.reduce_mean(net, axis=1, keep_dims=True)

        return net
    
    def layer_add(self, net, args, options):
        ops = self.ops
        gan = self.gan
        config = self.config

        orig = net
        if "layer" in options:
            net = self.layer(options["layer"])

        if len(args) > 0:
            if args[0] == 'noise':
                net = tf.random_normal(self.ops.shape(orig), mean=0, stddev=0.1)
            elif args[0] == 'layer_filter':
                net = config.layer_filter(gan, self.config, net)
            else:
                subnet = self.subnets[args[0]]
                for layer in subnet:
                    net = self.parse_layer(net, layer)
                    self.layers += [net]

        if 'mask' in options:
            options['activation']=tf.nn.sigmoid
            mask = self.layer_conv(orig, [self.ops.shape(net)[0]], options)
            if 'threshold' in options:
                mask = tf.greater(mask, float(options['threshold']))
                mask = tf.cast(mask, tf.float32)
            return mask * net + (1-mask)*orig
        if "lambda" in options:
            lam = self.parse_lambda(options)
            return orig + lam * net
        else:
            return orig + net

    def layer_conv_dts(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        net = self.do_ops_layer(self.ops.conv2d, net, int(args[0]*4), options)
        s = ops.shape(net)
        net = tf.depth_to_space(net, 2)

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

        net = tf.image.resize_images(net, [options.w or ops.shape(net)[1]*2, options.h or ops.shape(net)[2]*2],1)

        if options.concat:
            extra = self.layer(options.concat)
            if self.ops.shape(extra) != self.ops.shape(net):
                extra = tf.image.resize_images(extra, [self.ops.shape(net)[1],self.ops.shape(net)[2]], 1)
            net = tf.concat([net, extra], axis=len(self.ops.shape(net))-1)

        options = options.copy()
        options['stride'] = 1
        options['avg_pool'] = 1

        return self.do_ops_layer(self.ops.conv2d, net, int(args[0]), options)

    def layer_bicubic_conv(self, net, args, options):
        s = self.ops.shape(net)
        net = bicubic_interp_2d(net, [s[1]*2, s[2]*2])
        net = self.layer_conv(net, args, options)
        return net

    def get_conv_options(self, config, options):
        stride = options.stride or self.ops.config_option("stride", [1,1])
        fltr = options.filter or self.ops.config_option("filter", [3,3])
        avg_pool = options.avg_pool or self.ops.config_option("avg_pool", [1,1])

        if type(stride) != type([]):
            stride = [int(stride), int(stride)]

        if type(avg_pool) != type([]):
            avg_pool = [int(avg_pool), int(avg_pool)]

        if type(fltr) != type([]):
            fltr = [int(fltr), int(fltr)]
        return stride, fltr, avg_pool


    def layer_deconv(self, net, args, options):
        return self.do_ops_layer(self.ops.deconv2d, net, int(args[0]), options)

    def layer_conv_double(self, net, args, options):
        options["stride"] = 1
        options["avg_pool"] = 1
        x1 = self.layer_conv(net, args, options)
        y1 = self.layer_conv(net, args, options)
        x2 = self.layer_conv(net, args, options)
        y2 = self.layer_conv(net, args, options)
        x = tf.concat([x1,x2],axis=1)
        y = tf.concat([y1,y2],axis=1)
        net = tf.concat([x,y],axis=2)
        return net


    def layer_attention(self, net, args, options):
        ops = self.ops
        options = hc.Config(options)
        gan = self.gan
        if "lambda" in options:
            oj_lambda = options["lambda"]
        else:
            oj_lambda = 1.0
        c_scale = float(options.c_scale or 8)

        def _flatten(_net):
            return tf.reshape(_net, [ops.shape(_net)[0], -1, ops.shape(_net)[-1]])
        def _pool(_net, scale):
            ksize = [1,scale,1,1]
            _net = tf.nn.avg_pool(_net, ksize=ksize, strides=ksize, padding='SAME')
            return _net
        def _attn(_net, name=None):
            args = [ops.shape(_net)[-1]]
            name = name or self.ops.generate_name()
            options.name=name+'_fx'
            options['activation']=None
            options['avg_pool']=[1,1]
            fx = self.layer_conv(_net, args, options)
            options.name=name+'_gx'
            gx = self.layer_conv(_net, args, options)
            options.name=name+'_hx'
            hx = self.layer_conv(_net, args, options)
            if options.c_scale:
                c_scale
                fx = _pool(fx, c_scale)
                gx = _pool(gx, c_scale)
            bottleneck_shape = ops.shape(hx)
            fx = _flatten(fx)
            gx = _flatten(gx)
            hx = _flatten(hx)
            fx = tf.transpose(fx, [0,2,1])
            if options.dot_product_similarity:
                f = tf.matmul(gx,fx)
                bji = f / tf.cast(tf.shape(f)[-1], tf.float32)
            else:
                bji = tf.nn.softmax(tf.matmul(gx,fx))

            if options.h_activation:
                hx = ops.lookup(options.h_activation)(hx)
            oj = tf.matmul(bji, hx)
            oj = tf.reshape(oj, bottleneck_shape)
            #if options.final_conv:
            #args[0] = ops.shape(_net)[-1]
            #if options.final_activation == 'crelu':
            #    args[0] //= 2
            #options.name=name+'_oj'
            #oj = self.layer_conv(oj, args, options)
            if options.final_activation:
                oj = self.ops.lookup(options.final_activation)(oj)
            if options.enable_at_step:
                oj *= tf.cast(tf.greater(tf.train.get_global_step(),int(options.enable_at_step)), tf.float32)
            if options.only:
                return oj
            return oj


        ojs = [_attn(net, options.name) for i in range(self.config.heads or 1)]

        if options.concat:
            nets = [net] + [oj*oj_lambda for oj in ojs]
            return tf.concat(nets, axis=3)
        else:
            for oj in ojs:
                net += oj*oj_lambda
            return net



    def layer_conv_reshape(self, net, args, options):
        options = hc.Config(options)
        config = self.config
        ops = self.ops

        activation_s = options.activation or self.ops.config_option("activation")
        activation = self.ops.lookup(activation_s)
        depth = int(args[0])

        _, fltr, _ = self.get_conv_options(config, options)
        stride = [1, 1]

        initializer = None # default to global

        trainable = True
        if options.trainable == 'false':
            trainable = False
        net = ops.conv2d(net, fltr[0], fltr[1], stride[0], stride[1], depth*4, initializer=initializer, trainable=trainable)
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
    def layer_two_sample_stack(self, net, args, options):
        options = hc.Config(options)
        def _slice(_net):
            s = self.ops.shape(_net)
            s[0] = s[0] // 2
            _net1 = tf.slice(_net, [0,0,0,0], s)
            _net2 = tf.slice(_net, [s[0],0,0,0], s)
            return _net1, _net2
        net1, net2 = _slice(net)
        net1a, net1b = _slice(net1)
        net2a, net2b = _slice(net2)
        if options.mixup:
            alpha = tf.random_uniform([1], 0, 1)
            t1 = alpha * net1a + (1-alpha) * net1b
            t2 = alpha * net2a + (1-alpha) * net2b
            t1 = tf.reshape(t1, self.ops.shape(net1b))
            t2 = tf.reshape(t2, self.ops.shape(net2b))
        else:
            t1 = tf.concat([net1a, net1b], axis=3)
            t2 = tf.concat([net2a, net2b], axis=3)
        # hack fixes shape expectations
        #t1 = tf.concat([t1,t1], axis=0)
        #t2 = tf.concat([t2,t2], axis=0)
        target = tf.concat([t1, t2], axis=0)
        s = self.ops.shape(net)

        return target
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
        config = self.config
        if "stride" not in options:
            options["stride"]=1
        activation = options.activation or self.ops.config_option("activation")
        r = options.r or 2
        r = int(r)
        def _PS(X, r, n_out_channel):
                if n_out_channel >= 1:
                    bsize, a, b, c = X.get_shape().as_list()
                    bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
                    Xs=tf.split(X,r,3) #b*h*w*r*r
                    Xr=tf.concat(Xs,2) #b*h*(r*w)*r
                    X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c
                return X
        args[0]=depth*(r**2)
        if (activation == 'crelu' or activation == 'double_sided'):
            args[0] //= 2 
        y1 = self.layer_conv(net, args, options)
        ps = _PS(y1, r, depth)
        return ps

    def layer_tensorflowcv(self, net, args, options):
        from tensorflowcv.model_provider import get_model as tfcv_get_model
        from tensorflowcv.model_provider import init_variables_from_state_dict as tfcv_init_variables_from_state_dict
        pretrained = "pretrained" in args
        data_format = "channels_last"
        if not hasattr(tfcv, 'VERSION') or "hypergan" not in tfcv.VERSION:
            print("We use modified fork of tensorflowcv.  Clone https://github.com/HyperGAN/imgclsmob and run 'python setup.py develop'.")
            exit(-1)

        assert self.ops.shape(net)[1] == 224
        assert self.ops.shape(net)[2] == 224
        if not hasattr(self.gan, 'tensorflowcv'):
            self.gan.tensorflowcv = {}
            self.gan.tensorflowcv_weights = {}

        name=self.ops.generate_name()
        layer_index = -1
        if "layer_index" in options:
            layer_index = int(options["layer_index"])
        trainable = True
        if "trainable" in options:
            if options["trainable"].lower() == "false":
                trainable = False

        before = tf.trainable_variables()
        weights = None
        with tf.variable_scope(name, reuse=self.ops._reuse):
            if name in self.gan.tensorflowcv:
                tfcvnet = self.gan.tensorflowcv[name]
                weights = self.gan.tensorflowcv_weights[name]
            else:
                tfcvnet = tfcv_get_model(args[0], pretrained=pretrained, data_format="channels_last", layer_index=layer_index)
        self.gan.tensorflowcv[name]= tfcvnet
        result = tfcvnet(net)
        after = tf.trainable_variables()

        weights = list(set(after) - set(before))
        if len(weights) > 0:
            weights = list(set(after) - set(before))
            if trainable:
                self.gan.ops.weights += weights
                assert weights[0] in tf.trainable_variables()
                assert weights[0] in self.gan.variables()
            if pretrained:
                if not hasattr(self, "do_not_initialize"):
                    self.gan.do_not_initialize = []
                self.gan.do_not_initialize += weights
                assert weights[0] in self.gan.do_not_initialize
                tfcv_init_variables_from_state_dict(sess=self.gan.session, state_dict=tfcvnet.state_dict)
                print("Loaded pretrained weights: tensorflowcv " + args[0], len(weights))
            else:
                for w in weights:
                    self.gan.session.run(w.initializer)
                print("Initialized weights (not pretrained): tensorflowcv " + args[0], len(weights))

        self.gan.tensorflowcv_weights[name]= weights
        return result

    def layer_crop(self, net, args, options):
        options = hc.Config(options)
        if len(args) == 0:
            w = self.gan.width()
            h = self.gan.height()
            d = self.gan.channels()
        else:
            w = int(args[0])
            h = int(args[1])
            d = int(args[2])
        s = self.ops.shape(net)
        if w > s[1] or h > s[2] or d > s[3]:
            raise ConfigurationException("Input resolution too small for crop")
        net = tf.slice(net, [0,0,0,int(options.d_offset or 0)], [-1,w,h,d])
        return net

    def layer_noise(self, net, args, options):
        options = hc.Config(options)
        if "learned" in args or options.learned:
            channels = self.ops.shape(net)[-1]
            shape = [1,1,channels]
            initializer = None
            if options.initializer is not None:
                initializer = self.ops.lookup_initializer(options.initializer, options)
            trainable = True
            if "trainable" in options and options["trainable"] == "false":
                trainable = False
            with tf.variable_scope(self.ops.generate_name(), reuse=self.ops._reuse):
                weights = self.ops.get_weight(shape, 'B', initializer=initializer, trainable=trainable)
            net += tf.random_normal(self.ops.shape(net), stddev=0.1) * weights

        elif options.mask:
            options['activation']=tf.nn.sigmoid
            mask = self.layer_conv(net, args, options)
            net += tf.random_normal(self.ops.shape(net), stddev=0.1) * mask
        elif options.uniform:
            return tf.random_uniform(self.ops.shape(net), minval=-1, maxval=1, dtype=tf.float32)
        else:
            net += tf.random_normal(self.ops.shape(net), stddev=0.1)
        return net

    def layer_latent(self, net, args, options):
        return self.gan.latent.sample

    def layer_layer(self, net, args, options):
        options = hc.Config(options)
        if "src" in options:
            obj = getattr(self.gan, options.src)
        else:
            obj = self
        result = obj.layer(args[0])
        if result is None:
            print("Layer options: ", obj.named_layers.keys())
            raise ValidationException("layer "+args[0]+" not found.")
        return result
    def layer_layer_norm(self, net, args, options):
        return self.ops.lookup("layer_norm")(self, net)

    def layer_variational_noise(self, net, args, options):
        net *= tf.random_normal(self.ops.shape(net), mean=1, stddev=0.02)
        return net

    def layer_variational(self, net, args, options):
        ops = self.ops
        options['name']=self.ops.description+"k1"
        mu = self.layer_conv(net, args, options)
        options['name']=self.ops.description+"k2"
        sigma = self.layer_conv(net, args, options)
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        self.variational=[mu,sigma]
        if not self.ops._reuse:
            if(hasattr(self.gan, "variational")):
                self.gan.variational += [[mu,sigma]]
            else:
                self.gan.variational=[[mu,sigma]]
        return z


    def layer_concat_noise(self, net, args, options):
        noise = tf.random_normal(self.ops.shape(net), stddev=0.1)
        net = tf.concat([net, noise], axis=len(self.ops.shape(net))-1)
        return net

    def layer_concat(self, net, args, options):
        options = hc.Config(options)
        if len(args) > 0 and args[0] == 'noise':
            if options.type == 'uniform':
                extra = tf.random_uniform(self.ops.shape(net), -1, 1)
            else:
                extra = tf.random_normal(self.ops.shape(net), stddev=0.1)
        if 'layer' in options:
            extra = self.layer(options['layer'])

        if self.ops.shape(extra) != self.ops.shape(net):
            extra = tf.image.resize_images(extra, [self.ops.shape(net)[1],self.ops.shape(net)[2]], 1)

        if 'mask' in options:
            options['activation']=tf.nn.sigmoid
            mask = self.layer_conv(net, [self.ops.shape(net)[-1]], options)
            extra *= mask

        net = tf.concat([net, extra], axis=len(self.ops.shape(net))-1)
        return net

    def layer_gram_matrix(self, net, args, options):

        shape = self.ops.shape(net)
        num_channels = shape[3]

        bs = shape[0]
        net = tf.reshape(net, shape=[bs, -1, num_channels])

        net = tf.matmul(tf.transpose(net, perm=[0,2,1]), net)
        net = tf.reshape(net, shape=[bs, shape[1], shape[1], -1])

        return net

    def layer_minibatch(self, net, args, options):
        options = hc.Config(options)
        s = self.ops.shape(net)
        group_size = options.group_size or self.ops.shape(net)[0]
        group = tf.reshape(net, [group_size, -1, s[1], s[2], s[3]])
        group -= tf.reduce_mean(group, axis=0, keep_dims=True)
        group = tf.reduce_mean(tf.square(group), axis=0)
        group = tf.sqrt(group+1e-8)
        group = tf.reduce_mean(group, axis=[1,2,3], keep_dims=True)
        group = tf.tile(group, [group_size, s[1], s[2], s[3]])
        group = tf.concat([net, group], axis=3)
        return group

    def layer_relational(self, net, args, options):
        def concat_coor(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.ops.shape(net)[0], 1])
            o = tf.concat([o, tf.to_float(coor)], axis=1)
            return o


        # eq.1 in the paper
        # g_theta = (o_i, o_j, q)
        # conv_4 [B, d, d, k]
        d = net.get_shape().as_list()[1]
        all_g = []
        for i in range(d*d):
            o_i = net[:, int(i / d), int(i % d), :]
            o_i = concat_coor(o_i, i, d)
            for j in range(d*d):
                o_j = net[:, int(j / d), int(j % d), :]
                o_j = concat_coor(o_j, j, d)
                r_input = tf.concat([o_i, o_j], axis=1)
                if i == 0 and j == 0:
                    g_theta = self.gan.create_component(self.config.relational, name='relational', input=r_input, reuse=self.ops._reuse)
                    g_i_j = g_theta.sample
                    self.ops.weights += g_theta.variables()
                else:
                    g_i_j = g_theta.reuse(r_input)
                all_g.append(g_i_j)

        all_g = tf.stack(all_g, axis=0)
        all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
        return all_g

    def layer_pixel_norm(self, net, args, options):
        epsilon = 1e-8
        return net * tf.rsqrt(tf.reduce_mean(tf.square(net), axis=1, keepdims=True) + epsilon)

    def layer_double_resolution(self, net, args, options):

        def scale_up(piece):
            orig_shape = self.ops.shape(piece)
            orig_piece = piece
            piece = tf.reshape(piece, [-1, 1, 1])
            piece = tf.tile(piece, [1,2,2])
            ns = []
            squares = tf.reshape(piece, [-1, 2])
            cells = tf.split(squares, self.ops.shape(squares)[0], axis=0)
            for i in range(0,self.ops.shape(squares)[0],orig_shape[1]*2):
                ra = cells[i:(i+orig_shape[1]*2)]
                ns += ra[::2]+ra[1::2]
            ns = tf.concat(ns, axis=0)
            new_shape = [orig_shape[0], orig_shape[1]*2, orig_shape[2]*2, 1]
            result = tf.reshape(ns, new_shape)
            return result
        return scale_up(piece)
        #pieces = tf.split(net, self.ops.shape(net)[3], 3)
        #pieces = [scale_up(piece) for piece in pieces]
        #return tf.concat(pieces, axis=3)

    def layer_adaptive_instance_norm(self, net, args, options):
        if 'w' in options:
            f = self.layer(options['w'])
        else:
            f = self.layer('w')
        if f is None:
            raise("ERROR: Could not find named generator layer 'w', add name=w to the input layer in your generator")
        if len(args) > 0:
            w = args[0]
        else:
            w = 128
        #f2 = self.layer_linear(f, [w], options)
        opts = copy.deepcopy(dict(options))
        size = self.ops.shape(net)[3]
        if "activation" not in opts:
            opts["activation"]="null"
        feature = self.layer_linear(f, [size*2], opts)
        f1 = tf.reshape(self.ops.slice(feature, [0,0], [-1, size]), [-1, 1, 1, size])
        f2 = tf.reshape(self.ops.slice(feature, [0,size], [-1, size]), [-1, 1, 1, size])
        net = self.adaptive_instance_norm(net, f1,f2)
        return net

    def layer_adaptive_instance_norm2(self, net, args, options):
        if 'w' in options:
            f = self.layer(options['w'])
        else:
            f = self.layer('w')
        return self.adaptive_instance_norm2(net, f)

    def layer_zeros_like(self, net, args, options):
        return tf.zeros_like(net)

    def layer_const(self, net, args, options):
        options = hc.Config(options)
        s  = [1] + [int(x) for x in args[0].split("*")]
        trainable = True
        if "trainable" in options and options["trainable"] == "false":
            trainable = False
        initializer = None
        if "initializer" in options and options["initializer"] is not None:
            initializer = self.ops.lookup_initializer(options["initializer"], options)
        with tf.variable_scope(self.ops.generate_name(), reuse=self.ops._reuse):
            if options.mode=='softmax':
                possible = self.ops.get_weight(s, name='const', trainable=trainable, initializer=initializer)
                if len(s) == 4:
                    selected = self.do_ops_layer(self.ops.linear, net, str(s[3]), options)
                    selected = tf.reshape(selected, [self.gan.batch_size(),1,1,s[3]])
                    selected = tf.nn.softmax(selected, axis=3)
                    return possible * selected
                if len(s) == 5:
                    selected = self.do_ops_layer(self.ops.linear, net, str(s[4]), options)
                    selected = tf.reshape(selected, [self.gan.batch_size(),1,1,1,s[4]])
                    selected = tf.nn.softmax(selected, axis=4)
                    return tf.reduce_sum(possible * selected, axis=4)
            else:
                return tf.tile(self.ops.get_weight(s, name='const', trainable=trainable, initializer=initializer), [self.gan.batch_size(), 1,1,1])

    def layer_const_like(self, net, args, options):
        options = hc.Config(options)
        s = self.ops.shape(self.layer(args[0]))
        s[0] = 1
        trainable = True
        if "trainable" in options and options["trainable"] == "false":
            trainable = False
        initializer = None
        if "initializer" in options and options["initializer"] is not None:
            initializer = self.ops.lookup_initializer(options["initializer"], options)
        with tf.variable_scope(self.ops.generate_name(), reuse=self.ops._reuse):
            return tf.tile(self.ops.get_weight(s, name='const', trainable=trainable, initializer=initializer), [self.gan.batch_size(), 1,1,1])

    def layer_reduce_sum(self, net, args, options):
        net = tf.reduce_sum(net, axis=[1,2,3])
        return net



    def layer_reference(self, net, args, options):
        options = hc.Config(options)

        obj = self
        if "src" in options:
            obj = getattr(self.gan, options.src)
        if "resize_images" in options:
            if hasattr(obj, 'layer'):
                return self.layer_resize_images(obj.layer(options.name), options["resize_images"].split("*"), options)
            else:
                return self.layer_resize_images(getattr(obj, options.name), options["resize_images"].split("*"), options)
        else:
            return obj.layer(options.name)

    def layer_knowledge_base(self, net, args, options):
        if not hasattr(self, 'knowledge_base'):
            self.knowledge_base = []
        kb = tf.Variable(tf.zeros_like(net), dtype=tf.float32, trainable=False)
        self.knowledge_base.append([options['name'], kb])
        return tf.concat([net, kb], axis=-1)

    def layer_resize_images(self, net, args, options):
        options = hc.Config(options)
        if len(args) == 0:
            w = self.gan.width()
            h = self.gan.height()
        else:
            w = int(args[0])
            h = int(args[1])
        method = options.method or 1
        return tf.image.resize_images(net, [w, h], method=method)
    
    def layer_progressive_replace(self, net, args, options):
        start = self.layer(options["start"])
        end = self.layer(options["end"])
        steps = float(options["steps"])
        delay = 0
        if "delay" in options:
            delay = int(options["delay"])
        if self.ops.shape(start) != self.ops.shape(end):
            start = tf.image.resize_images(start, [self.ops.shape(end)[1], self.ops.shape(end)[2]],1)
        decay = (tf.cast(self.gan.steps, dtype=tf.float32)-tf.constant(delay, dtype=tf.float32)) / tf.constant(steps, dtype=tf.float32)

        decay = tf.minimum(1.0, decay)
        decay = tf.maximum(0.0, decay)
        self.gan.add_metric('decay', decay)
        self.gan.add_metric('gs', self.gan.steps)

        net = decay * end + (1.0-decay) * start
        if "name" in options:
            net = tf.identity(net, name=options["name"])
        return net

    def forward(self, x):
        return self.net(x).view(self.gan.batch_size(), -1)
