import tensorflow as tf
import hyperchamber as hc
import numpy as np

TINY=1e-12


def identity(config, gan, net):
  return net

def sphere(config, gan, net):
  net = gaussian(config, gan, net)
  spherenet = tf.square(net)
  spherenet = tf.reduce_sum(spherenet, 1)
  lam = tf.sqrt(spherenet+TINY)
  return net/tf.reshape(lam,[int(lam.get_shape()[0]), 1])

def modal(config, gan, net):
  net = tf.round(net*(config.modes))/(config.modes)
  return net

def binary(config, gan, net):
  net = tf.greater(net, 0)
  net = tf.cast(net, tf.float32)
  return net

def modal_gaussian(config, gan, net):
  a = modal(config, gan, net)
  b = gaussian(config, gan, net)
  return a + b * 0.1

def modal_sphere(config, gan, net):
  net = gaussian(config, gan, net)
  net = modal(config, gan, net)
  spherenet = tf.square(net)
  spherenet = tf.reduce_sum(spherenet, 1)
  lam = tf.sqrt(spherenet+TINY)
  return net/tf.reshape(lam,[int(lam.get_shape()[0]), 1])

def modal_sphere_gaussian(config, gan, net):
  net = modal_sphere(config, gan, net)
  return net + (gaussian(config, gan, net) * 0.01)

# creates normal distribution from uniform values https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
def gaussian(config, gan, net):
  z_dim = config.z
  net = (net + 1) / 2

  za = tf.slice(net, [0,0], [gan.config.batch_size, z_dim//2])
  zb = tf.slice(net, [0,z_dim//2], [gan.config.batch_size, z_dim//2])

  pi = np.pi
  ra = tf.sqrt(-2 * tf.log(za+TINY))*tf.cos(2*pi*zb)
  rb = tf.sqrt(-2 * tf.log(za+TINY))*tf.sin(2*pi*zb)

  return tf.reshape(tf.concat(axis=1, values=[ra, rb]), net.get_shape())


def periodic(config, gan, net):
  return periodic_triangle_waveform(net, config.periods)

def periodic_gaussian(config, gan, net):
  net = periodic_triangle_waveform(net, config.periods)
  return gaussian(config, gan, net)

def periodic_triangle_waveform(z, p):
  return 2.0 / np.pi * tf.asin(tf.sin(2*np.pi*z/p))

def config(z=[16,32,64],min=-1,max=1,projections=[[identity, modal, sphere]],
        modes=4):
  selector = hc.Selector()
  selector.set('create', create)
  selector.set('z', z)
  selector.set('min', min)
  selector.set('max', max)

  selector.set('projections', projections)
  selector.set('modes', modes)

  return selector.random_config()

def create(config, gan):
  zs = []
  z_base = tf.random_uniform([gan.config.batch_size, config.z],config.min, config.max,dtype=gan.config.dtype)
  for projection in config.projections:
      zs.append(projection(config, gan, z_base))
  zs = tf.concat(axis=1, values=zs)
  return zs, z_base

