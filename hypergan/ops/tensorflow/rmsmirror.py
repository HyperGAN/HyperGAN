#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.training.rmsprop import RMSPropOptimizer
import tensorflow as tf

# Adapted from https://raw.githubusercontent.com/openai/iaf/master/tf_utils/adamax.py

class RMSMirrorOptimizer(RMSPropOptimizer):


  def __init__(self, learning_rate=0.001, decay=0.9, momentum=0.0, p=0.1, epsilon=1e-8, use_locking=False, centered=False, name="RMSMirror", gan=None):
    super().__init__(learning_rate, decay=decay, momentum=momentum, epsilon=epsilon, use_locking=use_locking, centered=centered, name=name)
    self._p = p
    self.gan = gan

  def _prepare(self):
    super()._prepare()
    if self._p == "rand":
        self._p_t = tf.random_uniform([1], minval=0.0, maxval=1.0)
    else:
        self._p_t = ops.convert_to_tensor(self._p, name="p")

  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    # Create slots for the first and second moments.
    for v in var_list:
        if v in self.gan.d_vars():
          self._zeros_slot(v, "g", self._name)
      
  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    p_t = math_ops.cast(self._p_t, var.dtype.base_dtype)
    decay_t = math_ops.cast(self._decay_tensor, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)
    momentum_t = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype)

    if var in self.gan.d_vars():
        rms = self.get_slot(var, "rms")
        momentum = self.get_slot(var, "momentum")

        rms_t = rms.assign(decay_t*rms + (1-decay_t)*tf.square(grad))
        g_t = lr_t * grad / tf.sqrt(rms_t + epsilon_t)

        g_t_1 = self.get_slot(var, "g")
        g_t = g_t_1.assign( g_t )

        #movement = 2. * lr_t * g_t - lr_t * g_t_1
        movement = lr_t * g_t - p_t * lr_t * (g_t_1 - g_t)
        var_update = state_ops.assign_sub(var, movement)
        return control_flow_ops.group(*[var_update, rms_t, g_t])
    else:
        lr_g = self.gan.config.trainer.g_learn_rate or 0.25
        g_t = lr_g * grad
        movement = lr_g * g_t 
        var_update = state_ops.assign_sub(var, movement)
        return control_flow_ops.group(*[var_update])


  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
