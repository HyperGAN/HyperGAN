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

def dot(x, y):
    x = [tf.reshape(_x, [-1]) for _x in x]
    y = [tf.reshape(_y, [-1]) for _y in y]
    x = tf.concat(x, axis=0)
    y = tf.concat(y, axis=0)
    return tf.reduce_sum(tf.multiply(x,y))

def normalize(v, r, method):
  if(method == True):
      norm_factor = tf.sqrt(dot(state.p,state.p)) / (tf.sqrt(dot(v, v))+1e-32)
      v = [norm_factor * _h for _h in v]
  if(method == 2):
      norm_factor = tf.sqrt(tf.abs(dot(v,state.p))) / (tf.sqrt(dot(v, v))+1e-32)
      v = [norm_factor * _h for _h in v]
  if(method == 3):
      norm_factor = tf.sqrt(tf.abs(dot(r,r))) / (tf.sqrt(dot(v, v))+1e-32)
      v = [norm_factor * _h for _h in v]
  if(method == 4):
      r_mean = [tf.reduce_mean(tf.abs(_g)) for _g in r]
      v_mean = [tf.reduce_mean(tf.abs(_g)) for _g in v]
      r_mean = sum(r_mean)  / len(r_mean)
      v_mean = sum(v_mean)  / len(v_mean)
      norm_factor = r_mean / v_mean
      v = [norm_factor * _h for _h in v]
  if(method == 5):
      r_mean = [tf.reduce_mean(tf.abs(_g)) for _g in r]
      v_mean = [tf.reduce_mean(tf.abs(_g)) for _g in v]
      norm_factor = [_r / (_h+1e-12) for _r, _h in zip(r_mean, v_mean)]
      v = [_n * _h for _n, _h in zip(norm_factor, v)]
      norm_factor = sum(norm_factor) / len(norm_factor)

  if method == 6:
      def median(x):
        return tf.contrib.distributions.percentile(x, 50.0)
      r_mean = [median(tf.abs(_g)) for _g in r]
      v_mean = [median(tf.abs(_g)) for _g in v]
      norm_factor = [_r / (_h+1e-32) for _r, _h in zip(r_mean, v_mean)]
      v = [_n * _h for _n, _h in zip(norm_factor, v)]
      norm_factor = sum(norm_factor) / len(norm_factor)

  return v


class CompetitiveTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="CompetitiveTrainHook"):
      super().__init__(config=config, gan=gan, trainer=trainer, name=name)

  def step(self, i, nsteps, p, x_grads, y_grads, x_loss, y_loss, x_params, y_params):
      print("I >= ", i, nsteps)
      if i >= nsteps:
          return p

      hvp = self.hvp_function()
      lr = self.config.learn_rate or 1e-4
      if i == 0 and self.config.sga_lambda:
        # SGA term on rhs
        grad_rev = tf.gradients(x_loss, y_params)#, grad_ys=self.gan.loss.sample[1], stop_gradients=min_params)
        sga = hvp(x_loss, y_params, x_params, [lr * _g for _g in grad_rev], grads=grad_rev)
        sga = normalize(sga, p, self.config.normalize)
        p = [_p - (self.config.sga_lambda)*_s for _p, _s in zip(p, sga)]

      if self.config.reverse_loss:
          print("D %d %d %d %d" % (len(y_params), len(x_params), len(p), len(self.gan.d_vars())))
          h_1_v = hvp(y_loss, x_params, y_params, [lr * _p for _p in p])
          h_2_v = hvp(x_loss, y_params, x_params, [lr * _h for _h in h_1_v])
      else:
          print("D %d %d %d %d" % (len(y_params), len(x_params), len(p), len(self.gan.d_vars())))
          h_1_v = hvp(x_loss, x_params, y_params, [lr * _p for _p in p], grads=x_grads)
          h_2_v = hvp(y_loss, y_params, x_params, [lr * _h for _h in h_1_v], grads=y_grads)

      h_2_v = normalize(h_2_v, p, self.config.normalize)

      if self.config.force:
          p = h_2_v
      elif self.config.merge:
          p = [(1.0 - self.config.decay)*_p + self.config.decay*_h_2_v for _p, _h_2_v in zip(p, h_2_v)]
      else:
          p = [_p + (self.config.decay or 0.01)*_h_2_v for _p, _h_2_v in zip(p, h_2_v)]

      return self.step(i+1, nsteps, p, x_grads, y_grads, x_loss, y_loss, x_params, y_params)

  def gradients(self, d_grads, g_grads):
      nsteps = self.config.nsteps
      d_loss, g_loss = self.gan.loss.sample
      d_params = self.gan.d_vars()
      g_params = self.gan.g_vars()
      lr = self.config.learn_rate or 1e-4
      d_grads = self.step(0, nsteps, d_grads, d_grads, g_grads, d_loss, g_loss, d_params, g_params)
      g_grads = self.step(0, nsteps, g_grads, g_grads, d_grads, g_loss, d_loss, g_params, d_params)
      if self.config.final_hvp:
          hvp = self.hvp_function()
          d_grads2 = hvp(g_loss, g_params, d_params, [lr * _g for _g in g_grads])
          g_grads2 = hvp(d_loss, d_params, g_params, [lr * _d for _d in d_grads])
          d_grads2 = [_p + _g * (self.config.d_final_decay or self.config.final_decay or self.config.decay) for _p, _g in zip(d_grads, d_grads2)]
          g_grads2 = [_p + _g * (self.config.g_final_decay or self.config.final_decay or self.config.decay) for _p, _g in zip(g_grads, g_grads2)]
          d_grads = normalize(d_grads2, d_grads, self.config.normalize)
          g_grads = normalize(g_grads2, g_grads, self.config.normalize)

      return [d_grads, g_grads]

  def hvp_function(self):
      hvp = self.hvp
      mode = self.config.hvp
      if mode == 1:
          hvp = self.hvp1
      if mode == 2:
          hvp = self.hvp2
      if mode == 13:
          hvp = self.hvp13
      if mode == 15:
          hvp = self.hvp15
      return hvp
 
  def hvp1(self, ys, xs, xs2, vs, grads=None):
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      if grads is None:
          grads = tf.gradients(ys, xs)
      ones = [tf.ones_like(_v) for _v in xs]
      lop = tf.gradients(grads, xs, grad_ys=ones)
      rop = tf.gradients(lop, ones, grad_ys=vs)
      return tf.gradients(grads, xs2)

  def hvp2(self, ys, xs, xs2, vs, grads=None):
      if grads is None:
          grads = tf.gradients(ys, xs)
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      lop = tf.gradients(grads, xs2, grad_ys=vs)
      return tf.gradients(lop, vs)

  def hvp(self, ys, xs, xs2, vs, grads=None):
      if grads is None:
        grads = tf.gradients(ys, xs)
      grads = [_g * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      return tf.gradients(grads, xs2, grad_ys=vs)

  def hvp13(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        if grads is None:
            grads = tf.gradients(ys, xs)
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

        if grads is None:
            grads = tf.gradients(ys, xs)
        #grads = [_g + _v for _g, _v in zip(grads, vs)]
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in grads]
        ones = [tf.ones_like(_v) for _v in grads]
        lop = tf.gradients(grads, xs, grad_ys=ones)
        rop = tf.gradients(lop, ones, grad_ys=vs)
        rop = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in rop]
        rop = tf.gradients(rop, xs2)
        return rop

  def hvp15(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        if grads is None:
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



