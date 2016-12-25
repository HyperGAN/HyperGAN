#This encoder is random noise

import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *

def encode(config, x, y):
  z_dim = config['generator.z']
  #encoded_z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
  z_mu = None
  z_sigma = None
  z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
  set_tensor("z", z)
  z2 = tf.square(z)
  z3 = tf.ones_like(z)
  z4 = tf.exp(z)

  z = tf.concat(1, [z,z2,z3,z4])
  encoded_z = tf.zeros_like(z)
  return z, encoded_z, None, None


