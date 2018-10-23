#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import tensorflow as tf

# Adapted from https://raw.githubusercontent.com/openai/iaf/master/tf_utils/adamax.py

class GradientDescentMirrorOptimizer(GradientDescentOptimizer):


  def __init__(self, learning_rate=0.001, use_locking=False, name="GradientDescentMirror", p=0.1):
    super().__init__(learning_rate, use_locking=use_locking, name=name)
    self.p = p

  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "g", self._name)
      
  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)

    g_t = lr_t * grad

    g_t_1 = self.get_slot(var, "g")
    g_t = g_t_1.assign( g_t )

    #movement = 2. * lr_t * g_t - lr_t * g_t_1
    movement = lr_t * g_t - self.p * lr_t * (g_t_1 - g_t)
    var_update = state_ops.assign_sub(var, movement) #Adam would be lr_t * g_t
    return control_flow_ops.group(*[var_update, g_t])

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
