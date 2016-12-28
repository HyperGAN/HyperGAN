#This encoder is random noise

import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *

TINY=1e-12

def encode_gaussian(config, x, y):
  # creates normal distribution https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  z_dim = config['generator.z']
  z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
  set_tensor("z", z)
  z = (z + 1) / 2

  za = tf.slice(z, [0,0], [config['batch_size'], z_dim//2])
  zb = tf.slice(z, [0,z_dim//2], [config['batch_size'], z_dim//2])

  #w = tf.get_variable("g_encode_guassian", [z_dim//2], dtype=config['dtype'],
  #                      initializer=tf.random_normal_initializer(1, .02, dtype=config['dtype']))

  pi = np.pi
  ra = tf.sqrt(-2 * tf.log(za+TINY))*tf.cos(2*pi*zb)#*w)
  rb = tf.sqrt(-2 * tf.log(za+TINY))*tf.sin(2*pi*zb)#*w)

  za = za * 2 - 1
  zb = zb * 2 - 1

  z = tf.concat(1, [za,zb,ra,rb])
  encoded_z = tf.zeros_like(z)
  return z, encoded_z, None, None

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


