import hyperchamber as hc
import tensorflow as tf
import numpy as np
import types
import uuid
import importlib
import hypergan
from tensorflow.python.ops.variables import RefVariable
from hypergan.ops.tensorflow import layer_regularizers
from hypergan.ops.tensorflow.activations import lrelu, selu
from hypergan.ops.tensorflow.extended_ops import *
from hypergan.ops.tensorflow.sn import spectral_normed_weight
class TensorflowOps:
    def __init__(self, config={}, device="/gpu:0"):
        config = hc.Config(config)
        self.config = config
        dtype = config.dtype or "float32"
        initializer = self.config_option("initializer", "he_normal")

        self.dtype = self.parse_dtype(dtype)
        self.scope_count = 0
        self.description = ''
        self.weights = []
        self.biases = []
        self.device = config.device
        self.initialized = False
        self._reuse = False
        self.reuse_scope_count = 0
        self.reuse_context = 0
        self.initializer = self.lookup_initializer(initializer, config)

    def config_option(self, name, default=None, config=None):
        if config is None:
            config = self.config
        if name in config:
            return config[name]
        if "defaults" in config:
            if name in config["defaults"]:
                return config["defaults"][name]
        return default

    def lookup_initializer(self, name, config):
        if name == 'orthogonal':
            orthogonal_gain = self.config_option("orthogonal_gain", 1.0, config)
            return self.orthogonal_initializer(orthogonal_gain)
        elif name == 'he_normal':
            return self.he_normal_initializer()
        elif name == 'xavier':
            return self.xavier_initializer()
        elif name == 'stylegan':
            return self.stylegan_initializer(config or self.config)
        elif name == 'random_normal':
            return self.random_initializer(self.config_option("random_stddev", 0.02, config))
        else:
            raise Exception("initializer not found", name)

    def assert_tensor(self, net):
        if type(net) != tf.Tensor and type(net) != tf.Variable and type(net) != RefVariable:
            raise Exception("Expected a Tensor but received", net)

    def add_weights(self, weights):
        if not isinstance(weights, list):
            weights = [weights]
        self.weights += weights

    def variables(self):
        return self.biases + self.weights

    def random_initializer(self, stddev):
        def _build(shape):
            return tf.random_normal_initializer(0, stddev, dtype=self.dtype)
        return _build

    def stylegan_initializer(self, config):
        def _build(shape):
            gain = self.config_option("gain", np.sqrt(2), config)
            use_wscale = self.config_option("w_scale", False, config)
            lrmul = float(self.config_option("lrmul", 0.01, config))
            fan_in = np.prod([int(x) for x in shape[:-1]]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
            he_std = gain / np.sqrt(fan_in) # He init

            # Equalized learning rate and custom learning rate multiplier.
            if use_wscale:
                init_std = 1.0 / lrmul
                runtime_coef = he_std * lrmul
            else:
                init_std = he_std / lrmul
                runtime_coef = lrmul

            self.runtime_coef = runtime_coef

            # Create variable.
            return tf.initializers.random_normal(0, init_std)
        return _build

    def orthogonal_initializer(self, gain):
        def _build(shape):
            return tf.orthogonal_initializer(gain)
        return _build

    def he_normal_initializer(self):
        def _build(shape):
            return tf.variance_scaling_initializer()
        return _build

    def xavier_initializer(self):
        def _build(shape):
            return tf.contrib.layers.xavier_initializer()
        return _build

    def describe(self, description):
        self.description = description

    def reuse(self):
        self._reuse = True
        self.reuse_scope_count = 0
        self.reuse_context += 1

    def stop_reuse(self):
        self.reuse_context -= 1
        if self.reuse_context == 0:
            self._reuse = False

    def generate_scope(self):
        if self._reuse:
            self.reuse_scope_count += 1
            return str(self.reuse_scope_count)
        self.scope_count += 1
        return str(self.scope_count)

    def generate_name(self, name=None):
        if name == None:
            if self.description == "":
                return self.generate_scope()
            return self.description + "_" + self.generate_scope()
        else:
            return name

    def parse_dtype(self, dtype):
        if type(dtype) == Function:
            return dtype
        if dtype == 'float32':
            return tf.float32
        elif dtype == 'float16':
            return tf.float16
        else:
            raise Exception("dtype not defined: "+str(dtype))

    def get_weight(self, shape=None, name=None, initializer=None, trainable=None):
        if name == None:
            name = "w"
        if initializer == None:
            initializer = self.initializer
        initializer = initializer(shape)
        weight = tf.get_variable(name, shape, dtype=self.dtype, initializer=initializer, trainable=trainable)
        if not self._reuse:
            self.weights.append(weight)
        if hasattr(self, 'runtime_coef'):
            weight *= self.runtime_coef
            delattr(self, "runtime_coef") # todo, better way to pass variables from initialiszer
        return weight

    def get_bias(self, shape, constant=0.0, name=None, trainable=None):
        if name == None:
            name='b'
        bias = tf.get_variable(name, shape, initializer=tf.constant_initializer(constant, dtype=self.dtype), dtype=self.dtype, trainable=trainable)
        if not self._reuse:
            self.biases.append(bias)
        return bias
    
    def parse_dtype(self, dtype):
        if dtype == 'float32':
            return tf.float32
        elif dtype == 'float16':
            return tf.float16
        else:
            raise Exception("dtype not defined: "+str(dtype))

    def cosine_conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            w = self.get_weight([filter_h, filter_w, net.get_shape()[-1], output_dim])
            conv = tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')
            biases = self.get_bias([output_dim], 0.001)
            conv = tf.nn.bias_add(conv, biases)

            w_square = tf.square(w)
            #w_sum = tf.reduce_sum(w_square, [0,1,2])
            w_conv = tf.nn.conv2d(tf.ones_like(net), w_square, strides=[1, 1, 1, 1], padding='SAME')
            w_norm = tf.sqrt(w_conv + 1e-4)

            net_square = tf.square(net)
            w_ones = tf.ones_like(w)
            net_sum = tf.nn.conv2d(net_square, w_ones, strides=[1, 1, 1, 1], padding='SAME')
            net_norm = tf.sqrt(net_sum + 1e-4)

            return conv / (w_norm * net_norm)

    #def weightnorm_conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
    #    with tf.variable_scope(self.generate_name(), reuse=self._reuse):
    #        w = self.get_weight([filter_h, filter_w, net.get_shape()[-1], output_dim])
    #        g = self.get_weight(name='g', shape=[1,output_dim])
    #        conv = tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')
    #        b = self.get_bias([output_dim], 0.001)

    #        w_square = tf.square(w)
    #        #w_sum = tf.reduce_sum(w_square, [0,1,2])
    #        w_conv = tf.nn.conv2d(tf.ones_like(net), w_square, strides=[1, 1, 1, 1], padding='SAME')
    #        w_norm = tf.sqrt(w_conv + 1e-4)

    #        return (conv*g+b) / (w_norm)

    #def weightnorm_conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
    #    with tf.variable_scope(self.generate_name(), reuse=self._reuse):
    #        w = self.get_weight([filter_h, filter_w, net.get_shape()[-1], output_dim])
    #        g = self.get_weight(name='g', shape=[1,output_dim])
    #        b = self.get_bias([output_dim])

    #        # use weight normalization (Salimans & Kingma, 2016)
    #        W = tf.reshape(g,[1,1,1,output_dim])*tf.nn.l2_normalize(w,[0,1,2])

    #        # calculate convolutional layer output
    #        return tf.nn.bias_add(tf.nn.conv2d(net, W, [1, stride_h, stride_w, 1], padding='SAME'), b)

    def weightnorm_conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            # modified from https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
            # data based initialization of parameters
            g = self.get_weight(name='g', shape=[1,1,1,output_dim])#, initializer=scale_init)
            b = self.get_bias(shape=[output_dim])#, initializer=-m_init*scale_init)
            shape = [filter_h, filter_w, int(net.get_shape()[-1]),output_dim]
            V = self.get_weight(name='v', shape=shape)
            V_norm = tf.nn.l2_normalize(V, [0,1,2])
            x_init = tf.nn.conv2d(net, V_norm, [1, stride_h, stride_w, 1], padding="SAME")
            x_init = tf.nn.bias_add(x_init, b)
            m_init, v_init = tf.nn.moments(x_init, [0,1,2])
            scale_init = 1.0/tf.sqrt(v_init + 1e-8)
            x_init = tf.reshape(scale_init,[1,1,1,output_dim])*(x_init-tf.reshape(m_init,[1,1,1,output_dim]))
            return g*x_init

    def weightnorm2_conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim, padding="SAME"):
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            # modified from https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
            # data based initialization of parameters
            g = self.get_weight(name='g', shape=[1,1,1,output_dim])#, initializer=scale_init)
            shape = [filter_h, filter_w, int(net.get_shape()[-1]),output_dim]
            V = self.get_weight(name='v', shape=shape)
            V_norm = tf.nn.l2_normalize(V, [0,1,2])
            x_init = tf.nn.conv2d(net, V_norm, [1, stride_h, stride_w, 1], padding=padding)
            m_init, v_init = tf.nn.moments(x_init, [0,1,2])
            scale_init = 1.0/tf.sqrt(v_init + 1e-8)
            x_init = tf.reshape(scale_init,[1,1,1,output_dim])*(x_init-tf.reshape(m_init,[1,1,1,output_dim]))
            return g*x_init

    def weightnorm3_conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            # modified from https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
            # data based initialization of parameters
            g = self.get_weight(name='g', shape=[1,1,1,output_dim])#, initializer=scale_init)
            shape = [filter_h, filter_w, int(net.get_shape()[-1]),output_dim]
            V = self.get_weight(name='v', shape=shape)
            V_norm = tf.nn.l2_normalize(V, [0,1,2])*g
            x_init = tf.nn.conv2d(net, V_norm, [1, stride_h, stride_w, 1], padding="SAME")
            m_init, v_init = tf.nn.moments(x_init, [0,1,2])
            scale_init = 1.0/tf.sqrt(v_init + 1e-8)
            x_init = tf.reshape(scale_init,[1,1,1,output_dim])*(x_init-tf.reshape(m_init,[1,1,1,output_dim]))
            return x_init


    def spectralnorm_conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim, padding='SAME'):
        def spectral_norm(w, iteration=1):
           w_shape = w.shape.as_list()
           w = tf.reshape(w, [-1, w_shape[-1]])

           u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

           u_hat = u
           v_hat = None
           for i in range(iteration):
               v_ = tf.matmul(u_hat, tf.transpose(w))
               v_hat =tf.nn.l2_normalize(v_, [0,1])

               u_ = tf.matmul(v_hat, w)
               u_hat =tf.nn.l2_normalize(u_, [0,1])

           sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
           w_norm = w / sigma

           with tf.control_dependencies([u.assign(u_hat)]):
               w_norm = tf.reshape(w_norm, w_shape)

           return w_norm
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            w = self.get_weight([filter_h, filter_w, net.get_shape()[-1], output_dim])
            conv = tf.nn.conv2d(net, strides=[1, stride_h, stride_w, 1], padding=padding, filter=spectral_norm(w))
            biases = self.get_bias([output_dim])
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def weightnorm_deconv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            # modified from https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
            # data based initialization of parameters
            g = self.get_weight(name='g', shape=[1,1,1,output_dim])#, initializer=scale_init)
            b = self.get_bias(shape=[output_dim])#, initializer=-m_init*scale_init)
            shape = [filter_h, filter_w, output_dim, int(net.get_shape()[-1])]
            V = self.get_weight(name='v', shape=shape)
            V_norm = tf.nn.l2_normalize(V, [0,1,2])
            
            net_shape = self.shape(net)
            target_shape = [net_shape[0], net_shape[1]*stride_h, net_shape[2]*stride_w, output_dim]
            print(net, target_shape, V_norm)
            x_init = tf.nn.conv2d_transpose(net, V_norm, target_shape, [1, stride_h, stride_w, 1], padding="SAME")
            x_init = tf.nn.bias_add(x_init, b)
            m_init, v_init = tf.nn.moments(x_init, [0,1,2])
            scale_init = 1.0/tf.sqrt(v_init + 1e-8)
            x_init = tf.reshape(scale_init,[1,1,1,output_dim])*(x_init-tf.reshape(m_init,[1,1,1,output_dim]))
            return g*x_init


    def conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim, padding="SAME", initializer=None, name=None, trainable=True, bias=True):
        self.assert_tensor(net)


        layer_regularizer = self.config_option("layer_regularizer")
        if layer_regularizer == 'cosine_norm':
            return self.cosine_conv2d(net, filter_w, filter_h, stride_w, stride_h, output_dim)
        if layer_regularizer == 'weight_norm3':
            return self.weightnorm3_conv2d(net, filter_w, filter_h, stride_w, stride_h, output_dim)
        if layer_regularizer == 'weight_norm2':
            return self.weightnorm2_conv2d(net, filter_w, filter_h, stride_w, stride_h, output_dim, padding=padding)
        if layer_regularizer == 'weight_norm':
            return self.weightnorm_conv2d(net, filter_w, filter_h, stride_w, stride_h, output_dim)
        if layer_regularizer == 'spectral_norm':
            return self.spectralnorm_conv2d(net, filter_w, filter_h, stride_w, stride_h, output_dim, padding=padding)

        if self.config_option("l2_scale"):
            net = net / tf.sqrt(float(filter_w)/float(stride_w)*float(filter_h)/float(stride_h))

        with tf.variable_scope(self.generate_name(name), reuse=self._reuse):
            shape=[filter_h, filter_w, net.get_shape()[-1], output_dim]
            if initializer is None:
                initializer = self.initializer
            w = self.get_weight(shape, initializer=initializer, trainable=trainable)
            conv = tf.nn.conv2d(net, w, strides=[1, stride_h, stride_w, 1], padding=padding)
            if bias:
                biases = self.get_bias([output_dim], trainable=trainable)
                conv = tf.nn.bias_add(conv, biases)
            return conv

    def deconv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim, initializer=None, name=None, trainable=True, bias=True):
        self.assert_tensor(net)
        shape = self.shape(net)
        layer_regularizer = self.config_option("layer_regularizer")
        if layer_regularizer == 'weight_norm':
            return self.weightnorm_deconv2d(net, filter_w, filter_h, stride_w, stride_h, output_dim)
        output_shape = [shape[0], shape[1]*stride_h, shape[2]*stride_w, output_dim]
        init_bias = 0.
        with tf.variable_scope(self.generate_name(name), reuse=self._reuse):
            # filter : [height, width, output_channels, in_channels]
            shape=[filter_h, filter_w, output_dim, shape[3]]
            if initializer is None:
                initializer = self.initializer
            w = self.get_weight(shape, initializer=initializer, trainable=trainable)

            deconv = tf.nn.conv2d_transpose(net, w, output_shape=output_shape,
                                    strides=[1, stride_h, stride_w, 1])
            if bias:
                biases = self.get_bias([output_shape[-1]])
                deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            return deconv

    def cosine_linear(self, net, output_dim):
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            w = self.get_weight([self.shape(net)[1], output_dim], name='cos_w')
            b = self.get_bias([output_dim], constant=0.001)
            w_norm = tf.sqrt(tf.reduce_sum(w**2, axis=0, keep_dims=True) + b ** 2)+0.000001
            x_norm = tf.sqrt(tf.reduce_sum(net**2, axis=1, keep_dims=True) + 0.000001)
            return (tf.matmul(net, w) + 0.001 * b) / w_norm / x_norm

    def weight_norm_linear(self, net, output_dim):
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            g = self.get_weight([1, output_dim], name='weightnorm_g')
            v = self.get_weight([self.shape(net)[1], output_dim], name='weighnorm_v')
            v_norm = tf.nn.l2_normalize(v, [0])
            b = self.get_bias([output_dim], constant=0.001)
            return (tf.matmul(net, v_norm) * g+b)

    def linear(self, net, output_dim, initializer=None, name=None, trainable=True, bias=True):
        linear_type = self.config_option("linear_type")
        if linear_type == 'cosine':
            return self.cosine_linear(net, output_dim)
        if linear_type == 'weight_norm':
            return self.weight_norm_linear(net, output_dim)
        self.assert_tensor(net)
        shape = self.shape(net)
        with tf.variable_scope(self.generate_name(name), reuse=self._reuse):
            linshape=[shape[1], output_dim]
            if initializer is None:
                initializer = self.initializer
            w = self.get_weight(linshape, initializer=initializer, trainable=trainable)
            if(bias):
                bias = self.get_bias([output_dim], trainable=trainable)
                return tf.matmul(net, w) + bias
            else:
                return tf.matmul(net, w)

    def reduce_linear(self):
        def _build(net, axis=1):
            return self.linear(net, 1)
        return _build

    def nsoftplus(self, net):
        return tf.log(tf.exp(net)+1)/np.log(2) - 1.0

    def clamped(self, net):
        return tf.maximum(0., tf.minimum(net, 1.))

    def clamped_unit(self, net):
        return tf.maximum(-1., tf.minimum(net, 1.))

    def null(self):
        def _null(_x):
            return _x
        return _null

    def groupsort(self, n=2):
        def _activation(v):
            fv = tf.reshape(v,[-1])
            length = self.shape(fv)[0]
            if(length < n or length % n != 0):
                print("not sorting", length)
                return v
            fv = tf.reshape(v,[-1, n])
            if n == 2:
                b,c = tf.split(fv, 2, 1)
                newv = tf.concat([tf.minimum(b, c), tf.maximum(b,c)], axis=1)
                newv = tf.reshape(newv,self.shape(v))
                return newv

            newv = tf.contrib.framework.sort(fv)
            newv = tf.reshape(newv,self.shape(v))
            return newv

        return _activation


    def double_sided(self):
        def _activation(_x):
            activation = self.lookup(self.config_option("double_sided_activation", 'relu'))
            ops = self
            orig_shape = self.shape(_x)
            net = _x
            namesstr = self.activation_name
            names = None
            if namesstr is not None:
                names = namesstr.split(",")
            names = names or [None, None]

            if len(orig_shape) == 2:
                self.activation_name = names[0]
                a = activation(net)
                self.activation_name = names[1]
                b = activation(-net)
                net = tf.concat([a,b],axis=1)
            elif len(orig_shape) == 4:
                self.activation_name = names[0]
                a = activation(net)
                self.activation_name = names[1]
                b = activation(-net)
                net = tf.concat([a,b],axis=3)
            else:
                raise "Two sided relu activation requires input dimensions of 2 or 4"
            self.activation_name = namesstr
            return net


        return _activation



    def prelu(self):
        def _prelu(_x):
            orig_shape = self.shape(_x)
            _x = tf.reshape(_x, [orig_shape[0], -1])

            # TODO Hack, cant send through function params b/c must match tensorflow activations
            name = None
            trainable = None
            if hasattr(self, 'activation_name'):
                name = self.activation_name 

            if hasattr(self, 'activation_trainable'):
                if self.activation_trainable == 'false':
                    self.activation_trainable = False
                trainable = self.activation_trainable
            # /TODO

            name = name or self.generate_name()
            with tf.variable_scope(name, reuse=self._reuse):
                print("Creating variable",name,self._reuse, trainable)
                alphas = tf.get_variable('prelu', 
                          _x.get_shape()[-1],
                          initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                          dtype=tf.float32,
                          trainable=trainable)
                pos = tf.nn.relu(_x)
                neg = alphas * (_x - abs(_x)) * 0.5

            if not self._reuse:
                self.biases += [alphas]
            return tf.reshape(pos + neg, orig_shape)

        return _prelu

    def bipolar(self):
        def _bipolar(_x, name=None):
            activation = self.lookup(self.config_option("bipolar_activation", 'relu'))
            ops = self
            orig_shape = self.shape(_x)
            net = _x
            if len(orig_shape) == 2:
                a = tf.slice(net, [0,0], [ops.shape(net)[0], ops.shape(net)[1]//2])
                b = tf.slice(net, [0,ops.shape(net)[1]//2],[ops.shape(net)[0], ops.shape(net)[1]//2])
                a = activation(a)
                b = -activation(-b)
                net = tf.concat([a,b],axis=1)
            elif len(orig_shape) == 4:
                print("Size is", net)
                a = tf.slice(net, [0,0,0,0], [-1, -1,-1, ops.shape(net)[3]//2])
                b = tf.slice(net, [0,0,0,ops.shape(net)[3]//2],[-1, -1,-1,ops.shape(net)[3]//2])
                a = activation(a)
                b = -activation(-b)
                net = tf.concat([a,b],axis=3)
            else:
                raise "Bipolar activation requires input dimensions of 2 or 4"
            return tf.reshape(net, orig_shape)


        return _bipolar


    def swish(self, x):
        return x * tf.nn.sigmoid(x)

    def trelu(self):
        def _trelu(_x, name=None):
            activation = self.lookup(self.config_option("trelu_activation", 'relu'))
            orig_shape = self.shape(_x)
            _x = tf.reshape(_x, [orig_shape[0], -1])

            with tf.variable_scope(self.generate_name(name), reuse=self._reuse):
                alphas = tf.get_variable('trelu', 
                          _x.get_shape()[-1],
                          initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                          dtype=tf.float32)
                net = activation(_x - alphas) + alphas

            #TODO this is wrong - need to add to biases only on no reuse
            self.add_weights(alphas)
            return tf.reshape(net, orig_shape)

        return _trelu

    def gelu(self, x):
        return 0.5*x*(1+tf.nn.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))

    def frelu(self):
        def _frelu(_x, name=None):
            activation = self.lookup(self.config_option("frelu_activation", 'relu'))
            orig_shape = self.shape(_x)
            _x = tf.reshape(_x, [orig_shape[0], -1])

            with tf.variable_scope(self.generate_name(name), reuse=self._reuse):
                alphas = tf.get_variable('frelu', 
                          [1],
                          initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                          dtype=tf.float32)
                net = activation(_x) + alphas

            #TODO this is wrong - need to add to biases only on no reuse
            self.add_weights(alphas)
            return tf.reshape(net, orig_shape)

        return _frelu

    def reshape(self, net, shape):
        self.assert_tensor(net)
        return tf.reshape(net, shape)

    def concat(self, values=[], axis=0):
        return tf.concat(values=values, axis=axis)

    def resize_images(self, net, dims, op_type):
        self.assert_tensor(net)
        return tf.image.resize_images(net, dims, op_type)

    def slice(self, net, x, y):
        self.assert_tensor(net)
        return tf.slice(net, x, y)

    def shape(self, net):
        self.assert_tensor(net)
        return [(x._value or -1) for x in net.get_shape()]

    def add_n(self, net):
        return tf.add_n(net)

    def squash(self, net, reduce=tf.reduce_mean):
        """
        Takes any size tensor and reduces it to a single value using `reduce`.
        """
        while(sum(self.shape(net)) > 1):
            net = reduce(net)
            net = tf.squeeze(net)

        return net

    def lookup(self, symbol, use_eval=True):
        if symbol == None:
            return None

        if type(symbol) == type([]):
            return [self.lookup(k, use_eval=False) for k in symbol]

        if type(symbol) == type({}) or type(symbol) == hc.Config:
            return hc.Config({k: self.lookup(symbol[k], use_eval=False) for k in symbol.keys()})

        if type(symbol) != type(""):
            return symbol

        if not use_eval:
            return symbol

        if symbol.startswith('function:'):
            return self.lookup_function(symbol)

        if symbol.startswith('class:'):
            return self.lookup_class(symbol)

        if symbol == 'groupsort':
            return self.groupsort(self.config_option("groupsort_n", 2))
        if symbol == 'tanh':
            return tf.nn.tanh
        if symbol == 'sigmoid':
            return tf.nn.sigmoid
        if symbol == 'clamped':
            return self.clamped
        if symbol == 'clamped_unit':
            return self.clamped_unit
        if symbol == 'cosine_norm':
            return "cosine_norm"
        if symbol == 'batch_norm':
            return layer_regularizers.batch_norm_1
        if symbol == 'layer_norm':
            return layer_regularizers.layer_norm_1
        if symbol == "crelu":
            return tf.nn.crelu
        if symbol == 'null':
            return self.null()
        if symbol == "prelu":
            return self.prelu()
        if symbol == "double_sided":
            return self.double_sided()
        if symbol == 'nsoftplus':
            return self.nsoftplus
        if symbol == "trelu":
            return self.trelu()
        if symbol == "bipolar":
            return self.bipolar()
        if symbol == "swish":
            return self.swish
        if symbol == "selu":
            return selu
        if symbol == "frelu":
            return self.frelu()
        if symbol == "gelu":
            return self.gelu
        if symbol == "lrelu":
            return lrelu
        if symbol == "relu":
            return tf.nn.relu
        if symbol == 'square':
            return tf.square
        if symbol == 'reduce_mean':
            return tf.reduce_mean
        if symbol == 'reduce_min':
            return tf.reduce_min
        if symbol == 'reduce_sum':
            return tf.reduce_sum
        if symbol == 'reduce_logsumexp':
            return tf.reduce_logsumexp
        if symbol == 'reduce_linear':
            return self.reduce_linear()

        if symbol == 'l1_distance':
            return l1_distance
        if symbol == 'l2_distance':
            return l2_distance

        return symbol

    def lookup_function(self, name):
        namespaced_method = name.split(":")[1]
        method = namespaced_method.split(".")[-1]
        namespace = ".".join(namespaced_method.split(".")[0:-1])
        return getattr(importlib.import_module(namespace),method)

    def lookup_class(self, name):
        return self.lookup_function(name)

    def initialize_variables(self, session):
        with tf.device(self.device):
            if len(self.variables()) == 0:
                return
            init = tf.variables_initializer(self.variables())
            session.run(init)
            self.initialized = True

    def new_session(self, tfconfig):
        if tfconfig is None:
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            #tfconfig = tf.ConfigProto(log_device_placement=True)
            tfconfig.gpu_options.allow_growth=True

        with tf.device(self.device):
            return tf.Session(config=tfconfig)
