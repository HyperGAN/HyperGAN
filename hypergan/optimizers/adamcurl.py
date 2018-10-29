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

class CurlOptimizer(optimizer.Optimizer):


  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, p=0.01,
               use_locking=False, name="Adamcurl", gan=None):

    
    super(AdamcurlOptimizer, self).__init__(use_locking, name)
    self.gan = gan
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._p = p

  def _prepare(self):
    if self._p == "rand":
        self._p_t = tf.random_uniform([1], minval=0.0, maxval=1.0)
    else:
        self._p_t = ops.convert_to_tensor(self._p, name="p")



    self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")


  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "g", self._name)
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)
      self._get_or_make_slot(v, v, "w", self._name)
      
  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    p_t = math_ops.cast(self._p_t, var.dtype.base_dtype)
    if var.dtype.base_dtype == tf.float16:
        eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
    else:
        eps = 1e-8
    def movement_for(grad):
        v = self.get_slot(var, "v")
        v_t = v.assign(beta2_t * v + (1. - beta2_t) * tf.square(grad))
        m = self.get_slot(var, "m")
        m_t = m.assign( beta1_t * m + (1. - beta1_t) * grad )
        v_t_hat = tf.div(v_t, 1. - beta2_t)
        m_t_hat = tf.div(m_t, 1. - beta1_t)
        
        return [tf.div( m_t, tf.sqrt(v_t)+eps ), m_t, v_t]

    v = self.get_slot(var, "w")
    store_v = v.assign(var)
    movement, _, _ = movement_for(grad)
    movement *= lr_t
    var_update1 = state_ops.assign_sub(var, movement)

    if var in self.gan.d_vars():
        grad2 = tf.gradients(self.gan.trainer.d_loss, var)[0]
    elif var in self.gan.g_vars():
        grad2 = tf.gradients(self.gan.trainer.g_loss, var)[0]
    else:
        raise("Couldn't find var in g_vars or d_vars")

    grad3 = (grad - p_t * (grad2 - grad))

    movement2, m_t, v_t = movement_for(grad3)
    g_t_1 = self.get_slot(var, "g")
    g_t = g_t_1.assign( movement2 )

    reset_v = var.assign(v)
    var_update2 = var.assign_sub(movement2*lr_t, use_locking=self._use_locking)
    return control_flow_ops.group(*[store_v, var_update1, reset_v, var_update2, m_t, v_t, g_t])


    return cf
  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
