import math
from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

config = {
        }

def set_ops_globals(dtype, batch_size):
    config['dtype']=dtype
    config['batch_size']=batch_size

rng = np.random.RandomState([2016, 6, 1])

class layer_norm_1(object):
    def __init__(self, batch_size, name="layer_norm"):
        self.name = name
    def __call__(self, x, train=True):
        return tf.contrib.layers.layer_norm(x, scope=self.name, center=True, scale=True)


class batch_norm_1(object):
    """Code modification of http://stackoverflow.com/a/33950177

    """
    def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm", half=None):
        assert half is None
        del momentum # unused
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.batch_size = batch_size

            self.name=name

    def __call__(self, x, train=True):
        del train # unused

        shape = x.get_shape().as_list()

        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.random_normal_initializer(1., 0.02,dtype=config['dtype']))
            self.beta = tf.get_variable("beta", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.constant_initializer(0.,dtype=config['dtype']))

            self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])

            out =  tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

TRAIN_MODE = True
class conv_batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, name="batch_norm", epsilon=1e-5, momentum=0.1,
        in_dim=None):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
            self.in_dim = in_dim
            global TRAIN_MODE
            self.train = TRAIN_MODE
            self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
            print("initing %s in train: %s" % (scope.name, self.train))

    def __call__(self, x):
        shape = x.get_shape()
        shp = self.in_dim or shape[-1]
        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shp],dtype=config['dtype'],
                                         initializer=tf.random_normal_initializer(1., 0.02,dtype=config['dtype']))
            self.beta = tf.get_variable("beta", [shp],dtype=config['dtype'],
                                        initializer=tf.constant_initializer(0.,dtype=config['dtype']))

            self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])
            self.mean.set_shape((shp,))
            self.variance.set_shape((shp,))
            self.ema_apply_op = self.ema.apply([self.mean, self.variance])

            if self.train:
                # with tf.control_dependencies([self.ema_apply_op]):
                normalized_x = tf.nn.batch_norm_with_global_normalization(
                        x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                        scale_after_normalization=True)
            else:
                normalized_x = tf.nn.batch_norm_with_global_normalization(
                    x, self.ema.average(self.mean), self.ema.average(self.variance), self.beta,
                    self.gamma, self.epsilon,
                    scale_after_normalization=True)
            return normalized_x

class fc_batch_norm(conv_batch_norm):
    def __call__(self, fc_x):
        ori_shape = fc_x.get_shape().as_list()
        if ori_shape[0] is None:
            ori_shape[0] = -1
        new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
        x = tf.reshape(fc_x, new_shape)
        normalized_x = super(fc_batch_norm, self).__call__(x)
        return tf.reshape(normalized_x, ori_shape)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],dtype=config['dtype'],
                            initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=config['dtype']))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0, dtype=config['dtype']), dtype=config['dtype'])
        conv = tf.nn.bias_add(conv, biases)

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False,
             init_bias=0.):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=config['dtype'],
                            initializer=tf.random_normal_initializer(stddev=stddev, dtype=config['dtype']))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for versions of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], dtype=config['dtype'],initializer=tf.constant_initializer(init_bias, dtype=config['dtype']))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def special_deconv2d(input_, output_shape,
             k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False,
             init_bias=0.):
    # designed to reduce padding and stride artifacts in the generator

    # If the following fail, it is hard to avoid grid pattern artifacts
    assert k_h % d_h == 0
    assert k_w % d_w == 0

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],dtype=config['dtype'],
                            initializer=tf.random_normal_initializer(stddev=stddev,dtype=config['dtype']))

        def check_shape(h_size, im_size, stride):
            if h_size != (im_size + stride - 1) // stride:
                print( "Need h_size == (im_size + stride - 1) // stride")
                print( "h_size: ", h_size)
                print( "im_size: ", im_size)
                print( "stride: ", stride)
                print( "(im_size + stride - 1) / float(stride): ", (im_size + stride - 1) / float(stride))
                raise ValueError()

        check_shape(int(input_.get_shape()[1]), output_shape[1] + k_h, d_h)
        check_shape(int(input_.get_shape()[2]), output_shape[2] + k_w, d_w)

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[output_shape[0],
            output_shape[1] + k_h, output_shape[2] + k_w, output_shape[3]],
                                strides=[1, d_h, d_w, 1])
        deconv = tf.slice(deconv, [0, k_h // 2, k_w // 2, 0], output_shape)


        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias,dtype=config['dtype']),dtype=config['dtype'])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# http://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
prelu_count = 0
def prelu(prefix):
    def prelu_internal(_x):
        global prelu_count
        prelu_count += 1
        name = (prefix+"prelu_"+str(prelu_count))
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
    return prelu_internal


def sin_and_cos(x, name="ignored"):
    return tf.concat(len(x.get_shape()) - 1, [tf.sin(x), tf.cos(x)])

def maxout(x, k = 2):
    shape = [int(e) for e in x.get_shape()]
    ax = len(shape)
    ch = shape[-1]
    print('x',x,ch,k)
    assert ch % k == 0
    shape[-1] = ch // k
    shape.append(k)
    x = tf.reshape(x, shape)
    return tf.reduce_max(x, ax)

def offset_maxout(x, k = 2):
    shape = [int(e) for e in x.get_shape()]
    ax = len(shape)
    ch = shape[-1]
    print("--",ch,k)
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
    return tf.concat(dim, [lrelu(x), tf.minimum(tf.abs(x), tf.square(x))])

def linear(input_, output_size, scope=None, mean=0., stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], dtype=config['dtype'],
                                 initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=config['dtype']))
        bias = tf.get_variable("bias", [output_size],dtype=config['dtype'],
            initializer=tf.constant_initializer(bias_start,dtype=config['dtype']))
        if with_w:
            # import ipdb; ipdb.set_trace()
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

@contextmanager
def variables_on_cpu():
    old_fn = tf.get_variable
    def new_fn(*args, **kwargs):
        with tf.device("/cpu:0"):
            return old_fn(*args, **kwargs)
    tf.get_variable = new_fn
    yield
    tf.get_variable = old_fn

@contextmanager
def variables_on_gpu0():
    old_fn = tf.get_variable
    def new_fn(*args, **kwargs):
        with tf.device("/gpu:0"):
            return old_fn(*args, **kwargs)
    tf.get_variable = new_fn
    yield
    tf.get_variable = old_fn

def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

class batch_norm_second_half(object):
    """Code modification of http://stackoverflow.com/a/33950177

    """
    def __init__(self, epsilon=1e-5,  name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon

            self.name=name

    def __call__(self, x):

        shape = x.get_shape().as_list()

        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.random_normal_initializer(1., 0.02,dtype=config['dtype']))
            self.beta = tf.get_variable("beta", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.constant_initializer(0.,dtype=config['dtype']))

            second_half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])

            self.mean, self.variance = tf.nn.moments(second_half, [0, 1, 2])

            out =  tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

class batch_norm_first_half(object):
    """Code modification of http://stackoverflow.com/a/33950177

    """
    def __init__(self, epsilon=1e-5,  name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon

            self.name=name

    def __call__(self, x):

        shape = x.get_shape().as_list()

        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.random_normal_initializer(1., 0.02,dtype=config['dtype']))
            self.beta = tf.get_variable("beta", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.constant_initializer(0.,dtype=config['dtype']))

            first_half = tf.slice(x, [0, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])

            self.mean, self.variance = tf.nn.moments(first_half, [0, 1, 2])

            out =  tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

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

class batch_norm_cross(object):
    def __init__(self, epsilon=1e-5,  name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.name=name

    def __call__(self, x):

        shape = x.get_shape().as_list()

        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma0 = tf.get_variable("gamma0", [shape[-1] // 2],dtype=config['dtype'],
                                initializer=tf.random_normal_initializer(1., 0.02, dtype=config['dtype']))
            self.beta0 = tf.get_variable("beta0", [shape[-1] // 2],
                                initializer=tf.constant_initializer(0., dtype=config['dtype']))
            self.gamma1 = tf.get_variable("gamma1", [shape[-1] // 2],dtype=config['dtype'],
                                initializer=tf.random_normal_initializer(1., 0.02,dtype=config['dtype']))
            self.beta1 = tf.get_variable("beta1", [shape[-1] // 2],dtype=config['dtype'],
                                initializer=tf.constant_initializer(0.,dtype=config['dtype']))

            ch0 = tf.slice(x, [0, 0, 0, 0],
                              [shape[0], shape[1], shape[2], shape[3] // 2])
            ch1 = tf.slice(x, [0, 0, 0, shape[3] // 2],
                              [shape[0], shape[1], shape[2], shape[3] // 2])

            ch0b0 = tf.slice(ch0, [0, 0, 0, 0],
                                  [shape[0] // 2, shape[1], shape[2], shape[3] // 2])

            ch1b1 = tf.slice(ch1, [shape[0] // 2, 0, 0, 0],
                                  [shape[0] // 2, shape[1], shape[2], shape[3] // 2])


            ch0_mean, ch0_variance = tf.nn.moments(ch0b0, [0, 1, 2])
            ch1_mean, ch1_variance = tf.nn.moments(ch1b1, [0, 1, 2])

            ch0 =  tf.nn.batch_norm_with_global_normalization(
                ch0, ch0_mean, ch0_variance, self.beta0, self.gamma0, self.epsilon,
                scale_after_normalization=True)

            ch1 =  tf.nn.batch_norm_with_global_normalization(
                ch1, ch1_mean, ch1_variance, self.beta1, self.gamma1, self.epsilon,
                scale_after_normalization=True)

            out = tf.concat(3, [ch0, ch1])

            if needs_reshape:
                out = tf.reshape(out, orig_shape)

            return out

def constrained_conv2d(input_, output_dim,
           k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    assert k_h % d_h == 0
    assert k_w % d_w == 0
    # constrained to have stride be a factor of kernel width
    # this is intended to reduce convolution artifacts
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],dtype=config['dtype'],
                            initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=config['dtype']))

        # This is meant to reduce boundary artifacts
        padded = tf.pad(input_, [[0, 0],
            [k_h-1, 0],
            [k_w-1, 0],
            [0, 0]])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0,dtype=config['dtype']),dtype=config['dtype'])
        conv = tf.nn.bias_add(conv, biases)

        return conv

def masked_relu(x, name="ignored"):
    shape = [int(e) for e in x.get_shape()]
    prefix = [0] * (len(shape) - 1)
    most = shape[:-1]
    assert shape[-1] % 2 == 0
    half = shape[-1] // 2
    first_half = tf.slice(x, prefix + [0], most + [half])
    second_half = tf.slice(x, prefix + [half], most + [half])
    return tf.nn.relu(first_half) * tf.nn.sigmoid(second_half)


def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    print("RESHAPE", a, b,c,'--',a,b,r,r)
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  #
    bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
  # Main OP that you can arbitrarily use in you tensorflow code
  if color:
    Xc = tf.split(3, 3, X)
    X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
  else:
    X = _phase_shift(X, r)
  return X

