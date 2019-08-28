import numpy as np
import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# http://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
def prelu(prefix, i, _x):
    name = (prefix+"prelu_"+str(i))
    orig_shape = _x.get_shape()
    _x = tf.reshape(_x, [config['batch_size'], -1])

    #print("prelu for", _x.get_shape()[-1])
    alphas = tf.get_variable(name, 
            _x.get_shape()[-1],
            initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
            dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return tf.reshape(pos + neg, orig_shape)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def sin_and_cos(x, name="ignored"):
    return tf.concat(axis=len(x.get_shape()) - 1, values=[tf.sin(x), tf.cos(x)])

def maxout(x, k = 2):
    shape = [int(e) for e in x.get_shape()]
    ax = len(shape)
    ch = shape[-1]
    assert ch % k == 0
    shape[-1] = ch // k
    shape.append(k)
    x = tf.reshape(x, shape)
    return tf.reduce_max(x, ax)

rng = np.random.RandomState([2016, 6, 1])
def offset_maxout(x, k = 2):
    shape = [int(e) for e in x.get_shape()]
    ax = len(shape)
    ch = shape[-1]
    assert ch % k == 0
    shape[-1] = ch // k
    shape.append(k)
    x = tf.reshape(x, shape)
    ofs = rng.randn(1000, k).max(axis=1).mean()
    return tf.reduce_max(x, ax) - ofs

def lrelu_sq(x):
    """
    Concatenates lrelu and square
    """
    dim = len(x.get_shape()) - 1
    return tf.concat(axis=dim, values=[lrelu(x), tf.minimum(tf.abs(x), tf.square(x))])

def decayer(x, name="decayer"):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale", [1], initializer=tf.constant_initializer(1.,dtype=config['dtype']),dtype=config['dtype'])
        decay_scale = tf.get_variable("decay_scale", [1], initializer=tf.constant_initializer(1.,dtype=config['dtype']),dtype=config['dtype'])
        relu = tf.nn.relu(x)
        return scale * relu / (1. + tf.abs(decay_scale) * tf.square(decay_scale))

def decayer2(x, name="decayer"):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale", [int(x.get_shape()[-1])], initializer=tf.constant_initializer(1.,dtype=config['dtype']),dtype=config['dtype'])
        decay_scale = tf.get_variable("decay_scale", [int(x.get_shape()[-1])], initializer=tf.constant_initializer(1.,dtype=config['dtype']), dtype=config['dtype'])
        relu = tf.nn.relu(x)
        return scale * relu / (1. + tf.abs(decay_scale) * tf.square(decay_scale))

def masked_relu(x, name="ignored"):
    shape = [int(e) for e in x.get_shape()]
    prefix = [0] * (len(shape) - 1)
    most = shape[:-1]
    assert shape[-1] % 2 == 0
    half = shape[-1] // 2
    first_half = tf.slice(x, prefix + [0], most + [half])
    second_half = tf.slice(x, prefix + [half], most + [half])
    return tf.nn.relu(first_half) * tf.nn.sigmoid(second_half)

def minmax(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, -1)
    return net

def minmaxzero(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, 0)
    return net
