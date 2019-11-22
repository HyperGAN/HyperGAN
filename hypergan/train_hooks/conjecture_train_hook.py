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

class ConjectureTrainHook(BaseTrainHook):
  """ https://faculty.washington.edu/ratliffl/research/2019conjectures.pdf """
  def __init__(self, gan=None, config=None, trainer=None, name="ConjectureTrainHook"):
      super().__init__(config=config, gan=gan, trainer=trainer, name=name)

  def gradients(self, d_grads, g_grads):
      nsteps = self.config.nsteps
      d_loss, g_loss = self.gan.loss.sample
      d_params = self.gan.d_vars()
      g_params = self.gan.g_vars()
      lr = self.config.learn_rate or 1e-2

      if self.config.fast_conjectures:
          d_grads2 = self.hvp(g_loss, g_params, d_params, [lr * _g for _g in g_grads])
          g_grads2 = self.hvp(d_loss, d_params, g_params, [lr * _d for _d in d_grads])
          d_grads = [_p + _g * (self.config.fast_conjectures_gamma) for _p, _g in zip(d_grads, d_grads2)]
          g_grads = [_p + _g * (self.config.fast_conjectures_gamma) for _p, _g in zip(g_grads, g_grads2)]

      if self.config.fast_strategic_conjectures:
          f1 = d_loss
          f2 = g_loss
          p1 = d_params
          p2 = g_params
          d2f2 = tf.gradients(f2, p2)
          d1f2 = tf.gradients(f2, p1)
          d1f1 = tf.gradients(f1, p1)
          d2f1 = tf.gradients(f1, p2)

          d12f1d2f2 = self.hvp(f1, p2, p1, [lr * _g for _g in d2f2])
          d12f2d2f1 = self.hvp(f2, p2, p1, [lr * _g for _g in d2f1])

          d21f2d1f1 = self.hvp(f2, p1, p2, [lr * _g for _g in d1f1])
          d21f1d1f2 = self.hvp(f1, p1, p2, [lr * _g for _g in d1f2])
          d_grads = [_p - (_g1 + _g2) * (self.config.fast_strategic_conjectures_gamma) for _p, _g1, _g2 in zip(d_grads, d12f1d2f2, d12f2d2f1)]
          g_grads = [_p - (_g1 + _g2) * (self.config.fast_strategic_conjectures_gamma) for _p, _g1, _g2 in zip(g_grads, d21f1d1f2, d21f2d1f1)]

      return [d_grads, g_grads]

  def hvp(self, ys, xs, xs2, vs, grads=None):
      if grads is None:
        grads = tf.gradients(ys, xs)
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      return tf.gradients(grads, xs2, grad_ys=vs)
