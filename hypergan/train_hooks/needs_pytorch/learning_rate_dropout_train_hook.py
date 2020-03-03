from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from operator import itemgetter
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import hyperchamber as hc
import inspect
import numpy as np
import torch
import torch.nn as nn

class LearningRateDropoutTrainHook(BaseTrainHook):
  """ https://arxiv.org/abs/1912.00144 """
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.ones = self.gan.configurable_param(self.config.ones or 1.0)
      self.zeros = self.gan.configurable_param(self.config.zeros or 0.0)
      self.dropout = self.gan.configurable_param(self.config.dropout or 0.9)

  def forward(self):
      return [None, None]

  def gradients(self, d_grads, g_grads):
      d_ones = [torch.ones_like(_g) for _g in d_grads]
      g_ones = [torch.ones_like(_g) for _g in g_grads]
      d_zeros = [torch.zeros_like(_g) for _g in d_grads]
      g_zeros = [torch.zeros_like(_g) for _g in g_grads]
      da = [torch.where((torch.rand_like(_d_grads)- (1.0-self.dropout)) < 0, _o, _z) for _d_grads, _o, _z in zip(d_grads, d_ones, d_zeros)]
      ga = [torch.where((torch.rand_like(_g_grads)- (1.0-self.dropout)) < 0, _o, _z) for _g_grads, _o, _z in zip(g_grads, g_ones, g_zeros)]

      if self.config.skip_d is None:
          d_grads = [_a * _grad for _a, _grad in zip(da, d_grads)]
      if self.config.skip_g is None:
          g_grads = [_a * _grad for _a, _grad in zip(ga, g_grads)]
      #def count_params(variables):
      #  return np.sum([np.prod(self.ops.shape(t)) for t in variables ])
      #self.gan.add_metric('dropout-one', ones)
      #self.gan.add_metric('dropout-perc', dropout)
      #self.gan.add_metric('dropout', sum([self.ops.squash((1.0-_a/ones), tf.reduce_sum) for _a in da]) / count_params(da))

      return [d_grads, g_grads]
