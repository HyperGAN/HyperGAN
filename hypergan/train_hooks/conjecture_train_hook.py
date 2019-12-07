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
          if self.config.d_fast_conjectures_gamma != 0:
              d_grads = [_p + _g * (self.config.d_fast_conjectures_gamma or self.config.fast_conjectures_gamma) for _p, _g in zip(d_grads, d_grads2)]
          if self.config.g_fast_conjectures_gamma != 0:
              g_grads = [_p + _g * (self.config.g_fast_conjectures_gamma or self.config.fast_conjectures_gamma) for _p, _g in zip(g_grads, g_grads2)]

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
          if self.config.d_fast_strategic_conjectures_gamma != 0:
              d_grads = [_p - (_g1 + _g2) * (self.config.d_fast_strategic_conjectures_gamma or self.config.fast_strategic_conjectures_gamma) for _p, _g1, _g2 in zip(d_grads, d12f1d2f2, d12f2d2f1)]
          if self.config.g_fast_strategic_conjectures_gamma != 0:
              g_grads = [_p - (_g1 + _g2) * (self.config.g_fast_strategic_conjectures_gamma or self.config.fast_strategic_conjectures_gamma) for _p, _g1, _g2 in zip(g_grads, d21f1d1f2, d21f2d1f1)]

      if self.config.jare:
          # https://github.com/weilinie/JARE/blob/master/src/ops.py#L226
          d_grad_norm = tf.global_norm(d_grads)
          g_grad_norm = tf.global_norm(g_grads)
          d_reg = 0.5 * tf.square(g_grad_norm)
          g_reg = 0.5 * tf.square(d_grad_norm)
          d_grads2 = tf.gradients(d_reg, d_params)
          g_grads2 = tf.gradients(g_reg, g_params)
          if self.config.d_jare_gamma != 0:
              d_grads = [_p + _g * (self.config.d_jare_gamma or self.config.jare_gamma) for _p, _g in zip(d_grads, d_grads2)]
          if self.config.g_jare_gamma != 0:
              g_grads = [_p + _g * (self.config.g_jare_gamma or self.config.jare_gamma) for _p, _g in zip(g_grads, g_grads2)]

 
      if self.config.consensus:
          # https://github.com/weilinie/JARE/blob/master/src/ops.py#L127
          d_grad_norm = tf.global_norm(d_grads)
          g_grad_norm = tf.global_norm(g_grads)
          d_reg = 0.5 * tf.square(d_grad_norm)
          g_reg = 0.5 * tf.square(g_grad_norm)
          reg = d_reg + g_reg
          d_grads2 = tf.gradients(reg, d_params)
          g_grads2 = tf.gradients(reg, g_params)
          if self.config.d_consensus_gamma != 0:
              d_grads = [_p + _g * (self.config.d_consensus_gamma or self.config.consensus_gamma) for _p, _g in zip(d_grads, d_grads2)]
          if self.config.g_consensus_gamma != 0:
              g_grads = [_p + _g * (self.config.g_consensus_gamma or self.config.consensus_gamma) for _p, _g in zip(g_grads, g_grads2)]

      if self.config.sga:
          # SGA as defined in competitive gradient descent https://github.com/devzhk/Implicit-Competitive-Regularization/blob/master/optimizers.py 
          d_grad_rev = tf.gradients(g_loss, d_params)
          g_grad_rev = tf.gradients(d_loss, g_params)
          d_sga = self.hvp(d_loss, g_params, d_params, [lr * _g for _g in g_grad_rev])
          g_sga = self.hvp(g_loss, d_params, g_params, [lr * _g for _g in d_grad_rev])
          if self.config.d_sga_gamma != 0:
              d_grads = [_p - (self.config.d_sga_gamma or self.config.sga_gamma)*_s for _p, _s in zip(d_grads, d_sga)]
          if self.config.g_sga_gamma != 0:
              g_grads = [_p - (self.config.g_sga_gamma or self.config.sga_gamma)*_s for _p, _s in zip(g_grads, g_sga)]


      if self.config.dropout:
          # https://arxiv.org/abs/1912.00144
          mean = tf.constant(self.config.dropout_mean or 0.5)
          da = [tf.where((tf.random_uniform(self.ops.shape(_d_grads))- mean) < 0, tf.ones_like(_d_grads), tf.zeros_like(_d_grads)) for _d_grads in d_grads]
          ga = [tf.where((tf.random_uniform(self.ops.shape(_g_grads))- mean) < 0, tf.ones_like(_g_grads), tf.zeros_like(_g_grads)) for _g_grads in g_grads]
          d_grads = [_a * _grad for _a, _grad in zip(da, d_grads)]
          g_grads = [_a * _grad for _a, _grad in zip(ga, g_grads)]
          self.gan.add_metric('d_mask', sum([self.ops.squash(_a, tf.reduce_sum) for _a in da]) / (self.gan.parameter_count(da)))

      return [d_grads, g_grads]

  def losses(self):
      d_loss = self.gan.loss.sample[0]
      g_loss = self.gan.loss.sample[1]
      d_params = self.gan.d_vars()
      g_params = self.gan.g_vars()
      if self.config.locally_stable:
          d_gradient_norm_sq = tf.square(tf.global_norm(tf.gradients(g_loss, d_params)))
          self.d_loss = self.config.locally_stable_gamma * d_gradient_norm_sq
          self.gan.add_metric('locally_stable', self.d_loss)

      if self.config.stabilizing_training_js:
          # https://github.com/rothk/Stabilizing_GANs
          d1 = tf.nn.sigmoid(d_loss)
          d2 = tf.nn.sigmoid(g_loss)
          d1_grads = tf.gradients(d_loss, d_params)
          d2_grads = tf.gradients(g_loss, g_params)
          print(d1_grads)
          print("???")
          print(d_loss)
          d1_norm = [tf.norm(tf.reshape(_d1_grads, [-1])) for _d1_grads in d1_grads]
          d2_norm = [tf.norm(tf.reshape(_d2_grads, [-1])) for _d2_grads in d2_grads]

          reg_d1 = [tf.multiply(tf.square(1.0-d1), tf.square(_d1_norm)) for _d1_norm in d1_norm]
          reg_d2 = [tf.multiply(tf.square(d2), tf.square(_d2_norm)) for _d2_norm in d2_norm]
          reg_d1 = sum(reg_d1)
          reg_d2 = sum(reg_d2)
          self.d_loss = self.config.stabilizing_training_js_gamma * (reg_d1 + reg_d2)
          self.gan.add_metric('stable_js', self.ops.squash(self.d_loss))

      return [self.d_loss, self.g_loss]

  def hvp(self, ys, xs, xs2, vs, grads=None):
      if grads is None:
        grads = tf.gradients(ys, xs)
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      return tf.gradients(grads, xs2, grad_ys=vs)
