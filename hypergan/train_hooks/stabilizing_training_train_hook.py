#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class StabilizingTrainingTrainHook(BaseTrainHook):
  """ https://github.com/rothk/Stabilizing_GANs """
  def __init__(self, gan=None, config=None, trainer=None, name="StabilizingTrainingTrainHook"):
      super().__init__(config=config, gan=gan, trainer=trainer, name=name)
      self.d_loss = None
      self.g_loss = None

  def losses(self):
      d_loss = self.gan.loss.sample[0]
      g_loss = self.gan.loss.sample[1]
      d_params = self.gan.d_vars()
      g_params = self.gan.g_vars()

      d1 = tf.nn.sigmoid(d_loss)
      d2 = tf.nn.sigmoid(g_loss)
      d1_grads = tf.gradients(d_loss, d_params)
      d2_grads = tf.gradients(g_loss, g_params)
      d1_norm = [tf.norm(tf.reshape(_d1_grads, [-1])) for _d1_grads in d1_grads]
      d2_norm = [tf.norm(tf.reshape(_d2_grads, [-1])) for _d2_grads in d2_grads]

      reg_d1 = [tf.multiply(tf.square(1.0-d1), tf.square(_d1_norm)) for _d1_norm in d1_norm]
      reg_d2 = [tf.multiply(tf.square(d2), tf.square(_d2_norm)) for _d2_norm in d2_norm]
      reg_d1 = sum(reg_d1)
      reg_d2 = sum(reg_d2)
      self.d_loss = self.gan.configurable_param(self.config.gamma or 1.0) * (reg_d1 + reg_d2)
      self.gan.add_metric('stable_js', self.ops.squash(self.d_loss))

      return [self.d_loss, self.g_loss]
