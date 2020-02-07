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

class LearningRateDropoutTrainHook(BaseTrainHook):
  """ https://arxiv.org/abs/1912.00144 """
  def __init__(self, gan=None, config=None, trainer=None, name="LearnRateDropoutTrainHook"):
      super().__init__(config=config, gan=gan, trainer=trainer, name=name)

  def gradients(self, d_grads, g_grads):
      dropout = self.gan.configurable_param(self.config.dropout or 0.1)
      ones = self.gan.configurable_param(self.config.ones or 1.0)
      zeros = self.gan.configurable_param(self.config.zeros or 0.0)
      d_ones = [tf.fill(self.ops.shape(_g), ones) for _g in d_grads]
      g_ones = [tf.fill(self.ops.shape(_g), ones) for _g in g_grads]
      d_zeros = [tf.fill(self.ops.shape(_g), zeros) for _g in d_grads]
      g_zeros = [tf.fill(self.ops.shape(_g), zeros) for _g in g_grads]
      da = [tf.where((tf.random_uniform(self.ops.shape(_d_grads))- (1.0-dropout)) < 0, _o, _z) for _d_grads, _o, _z in zip(d_grads, d_ones, d_zeros)]
      ga = [tf.where((tf.random_uniform(self.ops.shape(_g_grads))- (1.0-dropout)) < 0, _o, _z) for _g_grads, _o, _z in zip(g_grads, g_ones, g_zeros)]
      if self.config.skip_d is None:
          d_grads = [_a * _grad for _a, _grad in zip(da, d_grads)]
      if self.config.skip_g is None:
          g_grads = [_a * _grad for _a, _grad in zip(ga, g_grads)]
      def count_params(variables):
        return np.sum([np.prod(self.ops.shape(t)) for t in variables ])
      self.gan.add_metric('dropout-one', ones)
      self.gan.add_metric('dropout-perc', dropout)
      self.gan.add_metric('dropout', sum([self.ops.squash((1.0-_a/ones), tf.reduce_sum) for _a in da]) / count_params(da))

      return [d_grads, g_grads]
