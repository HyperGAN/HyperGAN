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
import inspect
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class MemoryTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="MemoryTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    self.past_weights = []
    self.prev_sample = tf.Variable(self.gan.generator.sample, dtype=tf.float32)
    self.prev_zs = []
    print("PREV ", self.prev_zs)

  def losses(self):
    self.update_prev_sample = tf.assign(self.prev_sample, self.gan.generator.sample)
    self.prev_l2_loss = (self.config.prev_l2_loss_lambda or 0.1)*self.ops.squash(tf.square(self.gan.generator.sample-self.prev_sample))
    self.add_metric('prev_l2', self.prev_l2_loss)
    g_loss = self.prev_l2_loss * (self.config['lambda'] or 1)

    return [None, g_loss]

  def before_step(self, step, feed_dict):
    if step == 0:
        self.prev_zs = self.gan.session.run(self.gan.fitness_inputs(), feed_dict)

  def after_step(self, step, feed_dict):
    gan = self.gan
    gan.session.run(self.update_prev_sample)

    # assign prev sample for previous z
    # replace previous z with new z
    prev_feed_dict = {}
    for v, t in ( [ [v, t] for v, t in zip(self.prev_zs, gan.fitness_inputs())]):
        prev_feed_dict[t]=v
    # l2 = ||(pg(z0) - g(z0))||2
    prev_l2_loss = gan.session.run(self.prev_l2_loss, prev_feed_dict)
    # pg(z0) = g(z)
    self.prev_g = gan.session.run(self.update_prev_sample, feed_dict)
    # z0 = z
    self.prev_zs = gan.session.run(gan.fitness_inputs(), feed_dict)
    # optimize(l2, gl, dl)

    feed_dict[self.prev_l2_loss] = prev_l2_loss

