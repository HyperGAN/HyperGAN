import hyperchamber as hc
import tensorflow as tf
import types
import uuid
import importlib
import hypergan
from hypergan.ops.tensorflow import layer_regularizers
from hypergan.ops.tensorflow.activations import lrelu, selu
from hypergan.ops.tensorflow.extended_ops import *

class TensorflowOps:
    def __init__(self, config={}, device="/gpu:0"):
        config = hc.Config(config)
        dtype = config.dtype or "float32"
        initializer = config.initializer or 'orthogonal'
        orthogonal_gain = config.orthogonal_gain or 1.0
        random_stddev = config.random_stddev or 0.02

        self.dtype = self.parse_dtype(dtype)
        self.scope_count = 0
        self.description = ''
        self.weights = []
        self.biases = []
        self.device = config.device
        self.initialized = False
        self._reuse = False
        if initializer == 'orthogonal':
            self.initializer = self.orthogonal_initializer(orthogonal_gain)
        else:
            self.initializer = self.random_initializer(random_stddev)

    def assert_tensor(self, net):
        if type(net) != tf.Tensor and type(net) != tf.Variable:
            raise Exception("Expected a Tensor but received", net)

    def add_weights(self, weights):
        if not isinstance(weights, list):
            weights = [weights]
        self.weights += weights

    def variables(self):
        return self.biases + self.weights

    def random_initializer(self, stddev):
        def _build():
            return tf.random_normal_initializer(0, stddev, dtype=self.dtype)
        return _build

    def orthogonal_initializer(self, gain):
        def _build():
            return tf.orthogonal_initializer(gain)
        return _build

    def describe(self, description):
        self.description = description

    def reuse(self):
        self._reuse = True
        self.reuse_scope_count = 0

    def stop_reuse(self):
        self._reuse = False

    def generate_scope(self):
        if self._reuse:
            self.reuse_scope_count += 1
            return str(self.reuse_scope_count)
        self.scope_count += 1
        return str(self.scope_count)

    def generate_name(self):
        if self.description == "":
            return self.generate_scope()
        return self.description + "_" + self.generate_scope()

    def parse_dtype(self, dtype):
        if type(dtype) == Function:
            return dtype
        if dtype == 'float32':
            return tf.float32
        elif dtype == 'float16':
            return tf.float16
        else:
            raise Exception("dtype not defined: "+str(dtype))

    def describe(self, description):
        self.description = description

    def get_weight(self, shape):
        weight = tf.get_variable('w', shape, dtype=self.dtype, initializer=self.initializer())
        if not self._reuse:
            self.weights.append(weight)
        return weight

    def get_bias(self, shape):
        bias = tf.get_variable('b', shape, initializer=tf.constant_initializer(0.0, dtype=self.dtype), dtype=self.dtype)
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

    def conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        self.assert_tensor(net)
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            w = self.get_weight([filter_h, filter_w, net.get_shape()[-1], output_dim])
            conv = tf.nn.conv2d(net, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
            biases = self.get_bias([output_dim])
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def deconv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        self.assert_tensor(net)
        initializer = self.initializer()
        shape = self.shape(net)
        output_shape = [shape[0], shape[1]*stride_h, shape[2]*stride_w, output_dim]
        init_bias = 0.
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            # filter : [height, width, output_channels, in_channels]
            w = self.get_weight([filter_h, filter_w, output_dim, shape[3]])

            deconv = tf.nn.conv2d_transpose(net, w, output_shape=output_shape,
                                    strides=[1, stride_h, stride_w, 1])

            biases = self.get_bias([output_shape[-1]])
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            return deconv

    def linear(self, net, output_dim):
        self.assert_tensor(net)
        initializer = self.initializer()
        shape = self.shape(net)
        with tf.variable_scope(self.generate_name(), reuse=self._reuse):
            w = self.get_weight([shape[1], output_dim])
            bias = self.get_bias([output_dim])
            return tf.matmul(net, w) + bias

    def reduce_linear(self):
        def _build(net, axis=1):
            return self.linear(net, 1)
        return _build


    def prelu(self):
        def _prelu(_x):
            orig_shape = self.shape(_x)
            _x = tf.reshape(_x, [orig_shape[0], -1])

            with tf.variable_scope(self.generate_name(), reuse=self._reuse):
                alphas = tf.get_variable('prelu', 
                          _x.get_shape()[-1],
                          initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
                          dtype=tf.float32)
                pos = tf.nn.relu(_x)
                neg = alphas * (_x - abs(_x)) * 0.5

            self.add_weights(alphas)
            return tf.reshape(pos + neg, orig_shape)

        return _prelu

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

    def lookup(self, symbol):
        if symbol == None:
            return None

        if type(symbol) == type([]):
            return [self.lookup(k) for k in symbol]

        if type(symbol) == type({}) or type(symbol) == hc.Config:
            return hc.Config({k: self.lookup(symbol[k]) for k in symbol.keys()})

        if type(symbol) != type(""):
            return symbol

        if symbol.startswith('function:'):
            return self.lookup_function(symbol)

        if symbol.startswith('class:'):
            return self.lookup_class(symbol)

        if symbol == 'tanh':
            return tf.nn.tanh
        if symbol == 'sigmoid':
            return tf.nn.sigmoid
        if symbol == 'batch_norm':
            return layer_regularizers.batch_norm_1
        if symbol == 'layer_norm':
            return layer_regularizers.layer_norm_1
        if symbol == "crelu":
            return tf.nn.crelu
        if symbol == "prelu":
            return self.prelu()
        if symbol == "selu":
            return selu
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
            init = tf.variables_initializer(self.variables(), reuse=self._reuse)
            session.run(init)
            self.initialized = True

    def new_session(self, tfconfig):
        if tfconfig is None:
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            #tfconfig = tf.ConfigProto(log_device_placement=True)
            tfconfig.gpu_options.allow_growth=True

        with tf.device(self.device):
            return tf.Session(config=tfconfig)
