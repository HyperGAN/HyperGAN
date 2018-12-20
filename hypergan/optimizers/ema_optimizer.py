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
import inspect

class EmaOptimizer(optimizer.Optimizer):
  """Applies exponential moving average"""
  def __init__(self, learning_rate=0.001, decay=0.9, gan=None, config=None, use_locking=False, name="EmaOptimizer", optimizer=None):
    super().__init__(use_locking, name)
    self._decay = decay
    self.gan = gan
    self.config = config
    self.name = name
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
    g_vars = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
        elif var in self.gan.g_vars():
            g_vars += [var]
        else:
            raise("Couldn't find var in g_vars or d_vars")

    if self.config.apply_on == "discriminator":
        ema_vars = d_vars
    else:
        ema_vars = d_vars + g_vars
    with ops.init_scope():
        [self._get_or_make_slot(v, v, "ema", self._name) for v in ema_vars]
        self.optimizer._create_slots([v for g,v in grads_and_vars])
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                self._zeros_slot(var, "ema", self.name)

    self._prepare()
    ema_slots = [self.get_slot(v, "ema") for v in ema_vars]
    for name in self.optimizer.get_slot_names():
        for var in self.optimizer.variables():
            ema_vars += [var]
            ema_slots += [self._zeros_slot(var, "ema", self.name)]

    def calculate_ema(_v1,_v2):
        return self._decay *_v1 + (1-self._decay)*_v2
    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(ema_slots, ema_vars)]) # store variables
    with tf.get_default_graph().control_dependencies([op1]):
        op2 = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step, name=name)
        with tf.get_default_graph().control_dependencies([op2]):
            calculated_ema = [calculate_ema(v1, v2) for v1,v2 in zip(ema_slots, ema_vars)] # store variables
            op3 = tf.group(*[tf.assign(w, v) for w,v in zip(ema_vars, calculated_ema)])
            with tf.get_default_graph().control_dependencies([op3]):
                return tf.no_op()

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def variables(self):
      return super().variables() + self.optimizer.variables()
