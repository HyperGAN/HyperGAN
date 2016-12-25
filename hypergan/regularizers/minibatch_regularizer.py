import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *

# This is openai's implementation of minibatch regularization
def get_features(config,net):
  net = tf.reshape(net, [config['batch_size']*2, -1])
  minis= get_minibatch_features(config, net, config['batch_size']*2,config['dtype'])
  print("MINIS", minis)
  return minis

def get_minibatch_features(config, h,batch_size,dtype):
  single_batch_size = batch_size//2
  n_kernels = int(config['d_kernels'])
  dim_per_kernel = int(config['d_kernel_dims'])
  print("Discriminator minibatch is projecting from", h, "to", n_kernels*dim_per_kernel)
  x = linear(h, n_kernels * dim_per_kernel, scope="d_h")
  activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

  big = np.zeros((batch_size, batch_size))
  big += np.eye(batch_size)
  big = tf.expand_dims(big, 1)
  big = tf.cast(big,dtype=dtype)

  abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation,3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
  mask = 1. - big
  masked = tf.exp(-abs_dif) * mask
  def half(tens, second):
    m, n, _ = tens.get_shape()
    m = int(m)
    n = int(n)
    return tf.slice(tens, [0, 0, second * single_batch_size], [m, n, single_batch_size])
  # TODO: speedup by allocating the denominator directly instead of constructing it by sum
  #       (current version makes it easier to play with the mask and not need to rederive
  #        the denominator)
  f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
  f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

  return [f1, f2]



