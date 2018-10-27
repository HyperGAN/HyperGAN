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

  def __init__(self, learning_rate=0.001, use_locking=False, name="GradientDescentMirror", p=0.1, gan=None):
    super().__init__(learning_rate, use_locking=use_locking, name=name)
    self.gan = gan
    self._p = p

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
        #if v in self.gan.d_vars():
        #self._zeros_slot(v, "w", self._name)
        self._get_or_make_slot(v, v, "v", self._name)
    #self.sum = tf.zeros([1])
    #self.sum2 = tf.zeros([1])
      
  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    p_t = math_ops.cast(self._p_t, var.dtype.base_dtype)


    #if var in self.gan.d_vars():
    v = self.get_slot(var, "v")
    store_v = v.assign(var)
    #magnitude = tf.sqrt(tf.reduce_sum(tf.square(grad)))
    #magnitude_diff = tf.sqrt(tf.reduce_sum(tf.square(grad - g_t_1)))
    movement = lr_t * grad
    #movement = lr_t * (grad - p_t * (grad - g_t_1))# * (magnitude / magnitude_diff))
    #self.gan.add_metric('sum', magnitude_diff)
    #self.gan.add_metric('sum2', magnitude)
    #self.gan.add_metric('p', p_t)
    var_update1 = state_ops.assign_sub(var, movement)

    if var in self.gan.d_vars():
        grad2 = tf.gradients(self.gan.trainer.d_loss, var)[0]
    elif var in self.gan.g_vars():
        grad2 = tf.gradients(self.gan.trainer.g_loss, var)[0]
    else:
        raise("Couldn't find var in g_vars or d_vars")

    movement2 = lr_t * (grad - p_t * (grad2 - grad))# * (magnitude / magnitude_diff))
    #self.gan.add_metric('m1', tf.reduce_sum(movement))
    #self.gan.add_metric('m2', tf.reduce_sum(movement2))
    #self.gan.add_metric('diff', tf.reduce_sum(grad2-grad))
    reset_v = var.assign(v)
    var_update2 = state_ops.assign_sub(var, movement2)
    return control_flow_ops.group(*[store_v, var_update1, reset_v, var_update2])
    #else:
    #    movement = (self.gan.config.trainer.g_learn_rate or lr_t) * grad
    #    var_update = state_ops.assign_sub(var, movement)
    #    return control_flow_ops.group(*[var_update])


  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
