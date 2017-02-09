import tensorflow as tf
import numpy as np

TINY=1e-12

# creates normal distribution https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
def gaussian_from_uniform(config, gan, z):
  z_dim = gan.config.z
  z = (z + 1) / 2

  za = tf.slice(z, [0,0], [gan.config.batch_size, z_dim//2])
  zb = tf.slice(z, [0,z_dim//2], [gan.config.batch_size, z_dim//2])

  pi = np.pi
  ra = tf.sqrt(-2 * tf.log(za+TINY))*tf.cos(2*pi*zb)
  rb = tf.sqrt(-2 * tf.log(za+TINY))*tf.sin(2*pi*zb)

  return tf.reshape(tf.concat(1, [ra, rb]), z.get_shape())

# https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=2%2Fpi*arcsin(sin(2*pi*x%2Fy))
#
# p is periodicity
# amplitude is always amplitude of z (-1 to 1)
def periodic_triangle_waveform(z, p):
  return 2.0 / np.pi * tf.asin(tf.sin(2*np.pi*z/p))

