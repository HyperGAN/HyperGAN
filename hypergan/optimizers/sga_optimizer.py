# Symplectic Gradient Adjustment
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

class SgaOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="CurlOptimizer", optimizer=None, rho=1, beta=1, gamma=1,loss=None):
    super().__init__(use_locking, name)
    self._beta = beta
    self._rho = rho
    self._gamma = gamma
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate
    self.loss = loss
    optimizer["loss"] = loss

    self.optimizer = self.gan.create_optimizer(optimizer)
 
  def _prepare(self):
    super()._prepare()
    self.optimizer._prepare()

  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    self.optimizer._create_slots(var_list)

  def _apply_dense(self, grad, var):
    return self.optimizer._apply_dense(grad, var)


  def fwd_gradients(self, ys, xs, grad_xs=None, stop_gradients=None, colocate_gradients_with_ops=True):
    us = [tf.zeros_like(y) + float('nan') for y in ys]
    dydxs = tf.gradients(ys, xs, grad_ys=us,stop_gradients=stop_gradients,colocate_gradients_with_ops=colocate_gradients_with_ops)
    dydxs = [tf.zeros_like(x) if dydx is None else dydx for x,dydx in zip(xs,dydxs)]
    dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs, colocate_gradients_with_ops=colocate_gradients_with_ops)
    return dysdx
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    ws = [v for _,v in grads_and_vars]
    grads = [g for g,_ in grads_and_vars]
    self._prepare()

    jac_vec = self.fwd_gradients(grads,ws, grad_xs=grads,stop_gradients=ws)
    jac_vec = [tf.zeros_like(x) if dydx is None else dydx for x,dydx in zip(ws,jac_vec)]
    jac_tran_vec = tf.gradients(grads, ws, grad_ys=grads, stop_gradients=ws)
    jac_tran_vec = [tf.zeros_like(x) if dydx is None else dydx for x,dydx in zip(ws,jac_tran_vec)]
    at_xi = [(ht-h)*0.5 for (h,ht) in zip(jac_vec, jac_tran_vec)]


    if self.config.minus:
        new_grads = [g-a for g,a in zip(grads, at_xi)]
    else:
        new_grads = [g+a for g,a in zip(grads, at_xi)]
    grads_and_vars2 = zip(new_grads, ws)
    op8 = self.optimizer.apply_gradients(list(grads_and_vars2).copy(), global_step=global_step, name=name)
    with tf.get_default_graph().control_dependencies([op8]):
        return tf.no_op()


  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
  def variables(self):
      return super().variables() + self.optimizer.variables()
