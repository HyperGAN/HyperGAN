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
from tensorflow.python.ops.gradients_impl import _hessian_vector_product

class CompetitiveOptimizer(optimizer.Optimizer):
  """https://github.com/devzhk/Implicit-Competitive-Regularization/blob/master/optimizers.py ACGD"""
  def __init__(self, learning_rate=0.001, decay=0.9, gan=None, config=None, use_locking=False, name="CompetitiveOptimizer", optimizer=None):
    super().__init__(use_locking, name)
    self._decay = decay
    self.gan = gan
    self.config = config
    self.name = name
    self.learning_rate = learning_rate
    self.optimizer = self.gan.create_optimizer(optimizer)
 
  def _prepare(self):
    super()._prepare()
    self.optimizer._prepare()

  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    self.optimizer._create_slots(var_list)

  def _apply_dense(self, grad, var):
    return self.optimizer._apply_dense(grad, var)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    d_vars = []
    d_grads = []
    g_vars = []
    g_grads = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
            d_grads += [grad]
        elif var in self.gan.g_vars():
            g_vars += [var]
            g_grads += [grad]
        else:
            raise("Couldn't find var in g_vars or d_vars")
    min_params = d_vars
    max_params = g_vars
    grad_x = g_grads
    grad_y = d_grads
    lr = self.learning_rate

    grad_x_rev = tf.gradients(self.gan.loss.sample[0], max_params)
    grad_y_rev = tf.gradients(self.gan.loss.sample[1], min_params)

    hyp_x = self.hvpvec([_g*lr for _g in grad_x], max_params, grad_x_rev)
    hyp_y = self.hvpvec([_g*lr for _g in grad_y], min_params, grad_y_rev)

    self.gan.add_metric('hyp_x', sum([ tf.reduce_mean(_p) for _p in hyp_x]))
    self.gan.add_metric('hyp_y', sum([ tf.reduce_mean(_p) for _p in hyp_y]))

    rhs_x = [g - (self.config.sga_lambda or lr)*hyp for g, hyp in zip(grad_x, hyp_x)]
    rhs_y = [g - lr*hyp for g, hyp in zip(grad_y, hyp_y)]

    cg_y = self.conjugate_gradient(grad_x=rhs_y, grad_y=grad_y_rev,
            x_params=min_params, y_params=max_params, x=rhs_y, nsteps=(self.config.nsteps or 3),
            lr_x=self.learning_rate, lr_y=self.learning_rate)

    cg_x = self.conjugate_gradient(grad_x=rhs_x, grad_y=grad_x_rev,
            x_params=max_params, y_params=min_params, x=rhs_x, nsteps=(self.config.nsteps or 3),
            lr_x=self.learning_rate, lr_y=self.learning_rate)

    new_grad_x = cg_x
    new_grad_y = cg_y
    self.gan.add_metric('cg_x', sum([ tf.reduce_mean(_p) for _p in cg_x]))
    self.gan.add_metric('cg_y', sum([ tf.reduce_mean(_p) for _p in cg_y]))

    new_grads = new_grad_y + new_grad_x

    all_vars = d_vars + g_vars
    new_grads_and_vars = list(zip(new_grads, all_vars)).copy()
    return self.optimizer.apply_gradients(new_grads_and_vars)

  def conjugate_gradient(self, grad_x, grad_y, x_params, y_params, lr_x, lr_y, x, nsteps=10):
    for i in range(nsteps):
        h_1_v = self.hvpvec([_g*lr_x for _g in grad_x], y_params, [lr_x * _p for _p in x])
        h_1 = [lr_y * v for v in h_1_v]
        h_2 = self.hvpvec([_g*lr_y for _g in grad_y], x_params, h_1)
        x = [_h_2 + _x for _h_2, _x in zip(h_2, x)]
    return x

  def hvpvec(self, ys, xs, vs):
    result = self.fwd_gradients(ys, xs, stop_gradients=xs)
    result = [ r * v for r, v in zip(result, vs) ]
    return result

  def fwd_gradients(self, ys, xs, grad_xs=None, stop_gradients=None, colocate_gradients_with_ops=True, us=None):
    if us is None:
        us = [tf.zeros_like(y) + float('nan') for y in ys]
    dydxs = tf.gradients(ys, xs, grad_ys=us,stop_gradients=stop_gradients,colocate_gradients_with_ops=colocate_gradients_with_ops)
    dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs, colocate_gradients_with_ops=colocate_gradients_with_ops)
    return dysdx

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def variables(self):
      return super().variables() + self.optimizer.variables()
