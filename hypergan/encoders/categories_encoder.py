#This encoder is random multinomial noise

import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *

def random_category(self, batch_size, size, dtype):
    prior = tf.ones([batch_size, size])*1./size
    dist = tf.log(prior + TINY)
    with tf.device('/cpu:0'):
        sample=tf.multinomial(dist, num_samples=1)[:, 0]
        return tf.one_hot(sample, size, dtype=dtype)


def encode(config, x, y):
  z_dim = config['generator.z']
  encoded_z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
  categories = [self.random_category(config.batch_size, size, config.dtype) for size in config.categories]
  categories = tf.concat(1, categories)
  z_mu = None
  z_sigma = None
  z = categories
  graph.categories=categories
  return z, encoded_z, None, None

