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

class GradientPenaltyTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="GpSnMemoryTrainHook", memory_size=2, top_k=1):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    gan_inputs = self.gan.inputs.x
    self.s_max = [ tf.Variable( tf.zeros_like(gan_inputs)) for i in range(memory_size)]
    self.d_lambda = config['lambda'] or 1

    self.assign_s_max_new_entries = [ tf.assign(self.s_max[i], self.gan.sample_mixture()) for i in range(memory_size) ]
    self.memory_size = memory_size
    self.top_k = top_k

    gd = tf.gradients(gan.discriminator.sample, gan.d_vars())
    gds = [tf.square(tf.norm(_gd, ord=2)) for _gd in gd if _gd is not None]
    r = tf.add_n(gds)
    self.d_loss = self.d_lambda * tf.reduce_mean(r)
    self.gan.add_metric('gp', self.d_loss)

  def losses(self):
    return [self.d_loss, None]

  def after_step(self, step, feed_dict):
    pass

  def before_step(self, step, feed_dict):
    pass
