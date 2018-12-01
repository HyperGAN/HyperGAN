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

# Adapted from https://raw.githubusercontent.com/openai/iaf/master/tf_utils/adamax.py

class AdamirrorOptimizer(optimizer.Optimizer):


  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adamirror"):

    
    super(AdamirrorOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")


  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)
      self._zeros_slot(v, "g", self._name)
      
  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    if var.dtype.base_dtype == tf.float16:
        eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
    else:
        eps = 1e-8

    v = self.get_slot(var, "v")
    v_t = v.assign(beta2_t * v + (1. - beta2_t) * tf.square(grad))
    m = self.get_slot(var, "m")
    m_t = m.assign( beta1_t * m + (1. - beta1_t) * grad )
    v_t_hat = tf.div(v_t, 1. - beta2_t)
    m_t_hat = tf.div(m_t, 1. - beta1_t)
    
    g_t = tf.div( m_t, tf.sqrt(v_t)+eps )
    g_t_1 = self.get_slot(var, "g")
    g_t = g_t_1.assign( g_t )

    var_update = state_ops.assign_sub(var, 2. * lr_t * g_t - lr_t * g_t_1) #Adam would be lr_t * g_t
    return control_flow_ops.group(*[var_update, m_t, v_t, g_t])
  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
