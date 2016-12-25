import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *

# This is openai's implementation of minibatch regularization
def get_features(config,net):
  net = get_tensor('xgs')
  net = net[-1]
  #s = [int(x) for x in xg.get_shape()]
  #moments = tf.reshape(xg, [config['batch_size'], 2, s[1], s[2], s[3]])
  #moments = tf.nn.moments(xg, [1])
  #moments = tf.reshape(xg, s)
  #
  #result = tf.concat(3, [result, xg, moments])


  s = [int(x) for x in net.get_shape()]
  net1 = tf.slice(net, [0,0,0,0], [config['batch_size'], s[1], s[2], s[3]])
  net2 = tf.slice(net, [config['batch_size'],0,0,0], [config['batch_size'], s[1], s[2], s[3]])
  minisx = tf.reduce_mean(net1, reduction_indices=0, keep_dims=True)
  minisg = tf.reduce_mean(net2, reduction_indices=0, keep_dims=True)
  minis = tf.concat(0, [minisx, minisg])
  minis = tf.reshape(minis, [config['batch_size']*2, -1])
  print("MINIS", minis)
  return [minis]
