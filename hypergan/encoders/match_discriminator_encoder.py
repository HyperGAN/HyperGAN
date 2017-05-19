#This encoder is random multinomial noise

import tensorflow as tf
from hypergan.util.ops import *
import hypergan

import hyperchamber as hc

TINY = 1e-12

def config():
  selector = hc.Selector()
  selector.set('create', create)

  return selector.random_config()

def create(config, gan):
  with tf.variable_scope("measured", reuse=False):
      dconf = gan.config['discriminators'][0]
      dconf = hc.Config(hc.lookup_functions(dconf))
      print(dconf)
      discriminator = hypergan.discriminators.pyramid_discriminator.discriminator(gan, dconf, tf.zeros_like(gan.graph.x), tf.zeros_like(gan.graph.x), [], [], "d_")
      z_dim = int(discriminator.get_shape()[1])
  z = tf.random_uniform([gan.config.batch_size, z_dim],-1, 1,dtype=gan.config.dtype)
  return z, z

