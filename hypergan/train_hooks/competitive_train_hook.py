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

def dot(x, y):
    x = [tf.reshape(_x, [-1]) for _x in x]
    y = [tf.reshape(_y, [-1]) for _y in y]
    x = tf.concat(x, axis=0)
    y = tf.concat(y, axis=0)
    return tf.reduce_sum(tf.multiply(x,y))

class CompetitiveTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="CompetitiveTrainHook"):
      super().__init__(config=config, gan=gan, trainer=trainer, name=name)

  def step(self, i, nsteps, p, x_grads, y_grads, x_loss, y_loss, x_params, y_params):
      hvp = self.hvp
      if self.config.hvp == 1:
          hvp = self.hvp1
      if self.config.hvp == 2:
          hvp = self.hvp2
      if self.config.hvp == 13:
          hvp = self.hvp13
      if self.config.hvp == 15:
          hvp = self.hvp15
      if i >= nsteps:
          return p
      lr = self.config.learn_rate or 1e-4
      if self.config.hvp == 15 or self.config.hvp == 13:
          if self.config.reverse_loss:
              print("D %d %d %d %d" % (len(y_params), len(x_params), len(p), len(self.gan.d_vars())))
              h_1_v = hvp(y_loss, x_params, y_params, [lr * _p for _p in p])
              h_2_v = hvp(x_loss, y_params, x_params, [lr * _h for _h in h_1_v])
          else:
              print("D %d %d %d %d" % (len(y_params), len(x_params), len(p), len(self.gan.d_vars())))
              h_1_v = hvp(x_loss, x_params, y_params, [lr * _p for _p in p])
              h_2_v = hvp(y_loss, y_params, x_params, [lr * _h for _h in h_1_v])
      else:
          h_1_v = hvp(x_grads, x_params, y_params, [lr * _p for _p in p])
          h_2_v = hvp(y_grads, y_params, x_params, [lr * _h for _h in h_1_v])

      if(self.config.normalize == 5):
          rhs = p
          rhs_mean = [tf.reduce_mean(tf.abs(_g)) for _g in rhs]
          h_2_v_mean = [tf.reduce_mean(tf.abs(_g)) for _g in h_2_v]
          norm_factor = [_r / (_h+1e-32) for _r, _h in zip(rhs_mean, h_2_v_mean)]
          h_2_v = [_n * _h for _n, _h in zip(norm_factor, h_2_v)]
          norm_factor = sum(norm_factor) / len(norm_factor)
      else:
          norm_factor = tf.sqrt(dot(p,p)) / (tf.sqrt(dot(h_2_v, h_2_v))+1e-32)
          h_2_v = [norm_factor * _h for _h in h_2_v]

      if self.config.force:
          p = h_2_v
      else:
          p = [_p + (self.config.decay or 0.01)*_h_2_v for _p, _h_2_v in zip(p, h_2_v)]
      self.gan.add_metric("cnorm", norm_factor)

      return self.step(i+1, nsteps, p, x_grads, y_grads, x_loss, y_loss, x_params, y_params)

  def gradients(self, d_grads, g_grads):
      nsteps = self.config.nsteps or 5
      d_loss, g_loss = self.gan.loss.sample
      d_grads = self.step(0, nsteps, d_grads, d_grads, g_grads, d_loss, g_loss, self.gan.d_vars(), self.gan.g_vars())
      g_grads = self.step(0, nsteps, g_grads, g_grads, d_grads, g_loss, d_loss, self.gan.g_vars(), self.gan.d_vars())
      return [d_grads, g_grads]

  def hvp1(self, grads, x_params, y_params, vs):
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      ones = [tf.ones_like(_v) for _v in x_params]
      lop = tf.gradients(grads, x_params, grad_ys=ones)
      rop = tf.gradients(lop, ones, grad_ys=vs)
      return tf.gradients(grads, y_params)

  def hvp2(self, grads, x_params, y_params, vs):
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      lop = tf.gradients(grads, y_params, grad_ys=vs)
      return tf.gradients(lop, vs)

  def hvp(self, grads, x_params, y_params, vs):
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      return tf.gradients(grads, y_params, grad_ys=vs)

  def hvp13(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        grads = tf.gradients(ys, xs)
        #grads = [_g + _v for _g, _v in zip(grads, vs)]
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
        print("vS", vs)
        ones = [tf.ones_like(_v) for _v in grads]
        lop = tf.gradients(grads, xs, grad_ys=ones)
        rop = tf.gradients(lop, ones, grad_ys=vs)
        rop = tf.gradients(rop, xs2)
        print("ROP", rop)
        return rop

  def hvp14(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        grads = tf.gradients(ys, xs)
        #grads = [_g + _v for _g, _v in zip(grads, vs)]
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in grads]
        ones = [tf.ones_like(_v) for _v in grads]
        lop = tf.gradients(grads, xs, grad_ys=ones)
        rop = tf.gradients(lop, ones, grad_ys=vs)
        rop = tf.gradients(rop, xs2)
        return rop

  def hvp15(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        grads = tf.gradients(ys, xs)
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in grads]
        #grads = [_g + _v for _g, _v in zip(grads, vs)]
        print("vS", vs)
        ones = [tf.ones_like(_v) for _v in grads]
        lop = tf.gradients(grads, xs, grad_ys=ones)
        rop = tf.gradients(lop, ones)
        rop = tf.gradients(rop, xs2, vs)
        print("ROP", rop)
        return rop



