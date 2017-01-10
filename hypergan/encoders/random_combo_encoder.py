#This encoder is random noise

import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *

TINY=1e-12

# https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=2%2Fpi*arcsin(sin(2*pi*x%2Fy))
#
# p is periodicity
# amplitude is always amplitude of z (-1 to 1)
def periodic_triangle_waveform(config, z, p):
  return 2.0 / np.pi * tf.asin(tf.sin(2*np.pi*z/p))

# creates normal distribution https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
def gaussian_from_uniform(config, z):
  z_dim = config['generator.z']
  z = (z + 1) / 2

  za = tf.slice(z, [0,0], [config['batch_size'], z_dim//2])
  zb = tf.slice(z, [0,z_dim//2], [config['batch_size'], z_dim//2])

  pi = np.pi
  ra = tf.sqrt(-2 * tf.log(za+TINY))*tf.cos(2*pi*zb)
  rb = tf.sqrt(-2 * tf.log(za+TINY))*tf.sin(2*pi*zb)

  return ra, rb

def encode_periodic_gaussian(config, x, y):
  z_dim = config['generator.z']
  z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
  set_tensor("z", z)

  zgaus = tf.concat(1, gaussian_from_uniform(config, z))

  zs = [z, zgaus]

  for i in range(4):
    p = 2 ** (i+1)

    zi = periodic_triangle_waveform(config, z, p)
    zigaus = tf.concat(1, gaussian_from_uniform(config, zi))

    zs.append(zi)
    zs.append(zigaus)


  z = tf.concat(1, zs)
  encoded_z = tf.zeros_like(z)
  return z, encoded_z, None, None

def encode_multimodal_gaussian(config, x, y):
  # creates normal distribution https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  z_dim = config['generator.z']
  z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
  set_tensor("z", z)
  z = (z + 1) / 2

  za = tf.slice(z, [0,0], [config['batch_size'], z_dim//2])
  zb = tf.slice(z, [0,z_dim//2], [config['batch_size'], z_dim//2])


  pi = np.pi
  ra = tf.sqrt(-2 * tf.log(za+TINY))*tf.cos(2*pi*zb)#*w)
  rb = tf.sqrt(-2 * tf.log(za+TINY))*tf.sin(2*pi*zb)#*w)

  w = tf.get_variable("g_encode_cos", [z_dim], dtype=config['dtype'],
                        initializer=tf.random_normal_initializer(5, 1, dtype=config['dtype']))

  zsplitra = tf.sqrt(-2 * tf.log(za*1+TINY))*tf.cos(2*pi*zb)
  zsplitrb = tf.sqrt(-2 * tf.log(za*1+TINY))*tf.sin(2*pi*zb)

  zcosa = tf.cos(2*pi*za)
  zcosb = tf.cos(2*pi*zb)
  zsina = tf.sin(2*pi*za)
  zsinb = tf.sin(2*pi*zb)

  za = za * 2 - 1
  zb = zb * 2 - 1

  z = tf.concat(1, [za,zb,ra,rb, zcosa, zcosb, zsina, zsinb, zsplitra, zsplitrb])
  encoded_z = tf.zeros_like(z)
  return z, encoded_z, None, None


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


