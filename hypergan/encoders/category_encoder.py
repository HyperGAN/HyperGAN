#This encoder is random multinomial noise

import tensorflow as tf

import hyperchamber as hc

from hypergan.encoders.base_encoder import BaseEncoder

TINY = 1e-12

class CategoryEncoder(BaseEncoder):
    def required(self):
        return "categories".split()

    def create(self):
        gan = self.gan
        ops = self.ops
        config = self.config

        categories = [self.random_category(gan.batch_size(), size, ops.dtype) for size in config.categories]
        self.categories = categories
        categories = tf.concat(axis=1, values=categories)
        self.sample = categories
        return categories

    def random_category(self, batch_size, size, dtype):
        prior = tf.ones([batch_size, size])*1./size
        dist = tf.log(prior + TINY)
        sample=tf.multinomial(dist, num_samples=1)[:, 0]
        return tf.one_hot(sample, size, dtype=dtype)
