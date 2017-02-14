import tensorflow as tf
import hyperchamber as hc
import numpy as np

TINY=1e-12

def config():
  selector = hc.Selector()
  selector.set('create', create)
  selector.set('min', -1)
  selector.set('max', 1)

  selector.set('projections', [[periodic, periodic_gaussian, gaussian]])
  selector.set('periods', 4)

  return selector.random_config()

def create(config, gan):
  zs = []
  z_base = tf.random_uniform([gan.config.batch_size, gan.config.z],config.min, config.max,dtype=gan.config.dtype)
  gan.graph.z.append(z_base)
  zs.append(z_base)
  for projection in config.projections:
      zs.append(projection(config, gan, z_base))
  zs = tf.concat(1, zs)
  return zs, z_base

def periodic(config, gan, net):
  return periodic_triangle_waveform(net, config.periods)

def periodic_gaussian(config, gan, net):
  net = periodic_triangle_waveform(net, config.periods)
  return gaussian(config, gan, net)

# creates normal distribution from uniform values https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
def gaussian(config, gan, z):
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

