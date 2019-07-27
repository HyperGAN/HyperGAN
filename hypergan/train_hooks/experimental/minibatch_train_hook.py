#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class MinibatchTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="MinibatchTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)

    config = self.config
    setattr(gan,config.name or 'minibatch', self)

    net = gan.discriminator.layer(config.target or 'minibatch')
    minibatch_size = gan.batch_size()*2
    single_batch_size = gan.batch_size()
    if self.config.gan_input is None:
        n_kernels = config.kernels or 256
        dim_per_kernel = config.dim_per_kernel or 10
        print("[discriminator] minibatch from", net, "to", n_kernels*dim_per_kernel)
        net = tf.reshape(net, [minibatch_size, -1])
        initializer = self.ops.lookup_initializer("stylegan", config.initializer_options or {})
        x = self.ops.linear(net, n_kernels * dim_per_kernel, initializer=initializer)
        activation = tf.reshape(x, (minibatch_size, n_kernels, dim_per_kernel))

        big = np.zeros((minibatch_size, minibatch_size))
        big += np.eye(minibatch_size)
        big = tf.expand_dims(big, 1)
        big = tf.cast(big,dtype=tf.float32)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation,3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask
        def half(tens, second):
            m, n, _ = tens.get_shape()
            m = int(m)
            n = int(n)
            return tf.slice(tens, [0, 0, second * single_batch_size], [m, n, single_batch_size])

        f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
        f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))
        stack_f = tf.concat([f1,f2], axis=0)

    if self.config.type == 'gan':
        if self.config.gan_input == 'distance':
            real, fake = tf.split(net, 2, 0)
            stack_f = tf.concat([real - fake,fake-real],  axis=0)
        if self.config.gan_input == 'moments':
            real, fake = tf.split(net, 2, 0)
            rm,rv = tf.nn.moments(real, 0)
            fm,fv = tf.nn.moments(fake, 0)
            rm = tf.expand_dims(rm,0)
            rv = tf.expand_dims(rv,0)
            fm = tf.expand_dims(fm,0)
            fv = tf.expand_dims(fv,0)
            print("_____", fm)
            r = tf.concat([rm, rv], axis=0)
            f = tf.concat([fm, fv], axis=0)
            stack_f = tf.concat([r,f],  axis=0)
        minibatch_discriminator = gan.create_component(self.config.discriminator or gan.config.minibatch_discriminator, name=(config.name or 'minibatch_discriminator'), input=stack_f)
        setattr(gan, config.name+"_discriminator", minibatch_discriminator)
        gan.discriminator.add_variables(minibatch_discriminator)
        loss = gan.create_component(self.gan.config.loss, minibatch_discriminator)
        gan.add_metric('mini_dl', loss.sample[0])
        gan.add_metric('mini_gl', loss.sample[1])
        self.loss = loss.sample
    else:
        self.minibatch_value = tf.concat([f1, f2], axis=1)
        self.minibatch_feed = gan.discriminator.layer('minibatch_feed')
        self.loss = [None, None]
    gan.discriminator.add_variables(self)

  def losses(self):
      return self.loss

  def after_step(self, step, feed_dict):
      pass

  def before_step(self, step, feed_dict):
    if self.config.type != 'gan':
      feed_dict[self.minibatch_feed]=self.gan.session.run(self.minibatch_value)

    pass
