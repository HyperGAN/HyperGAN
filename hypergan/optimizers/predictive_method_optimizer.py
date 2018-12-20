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

class PredictiveMethodOptimizer(optimizer.Optimizer):
  """ https://openreview.net/pdf?id=Skj8Kag0Z """
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="PredictiveMethodOptimizer", optimizer=None, rho=1, beta=1, gamma=1):
    super().__init__(use_locking, name)
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate

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
    var_list = [ v for _,v in grads_and_vars]
    d_vars = []
    g_vars = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
        elif var in self.gan.g_vars():
            g_vars += [var]
        else:
            raise("Couldn't find var in g_vars or d_vars")

    with ops.init_scope():
        v1 = [self._zeros_slot(v, "v1", self._name) for _,v in grads_and_vars]
        if self.config.include_slots:
            for name in self.optimizer.get_slot_names():
                for var in self.optimizer.variables():
                    self._zeros_slot(var, "pm", "pm")
    self._prepare()

    v1 = [self.get_slot(v, "v1") for _,v in grads_and_vars]
    slots_list = []
    slots_vars = []
    if self.config.include_slots:
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                slots_vars += [var]
                slots_list.append(self._zeros_slot(var, "pm", "pm"))


    current_vars = var_list + slots_vars
    tmp_vars = v1 + slots_list
    all_grads = [ g for g, _ in grads_and_vars ]

    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(tmp_vars, current_vars)]) # store variables

    with tf.get_default_graph().control_dependencies([op1]):
        # store g2
        #op3 = tf.group(*[tf.assign_sub(v, self._lr_t*grad) for grad,v in grads_and_vars])
        op3 = self.optimizer.apply_gradients(grads_and_vars.copy(), global_step=global_step, name=name)
        with tf.get_default_graph().control_dependencies([op3]):

            def pmcombine(_v1,_v2):
                return _v2 + (_v2 - _v1)

            combined = [pmcombine(_v1, _v2) for _v1, _v2 in zip(tmp_vars, current_vars)]
            # restore v1, slots
            op5 = tf.group(*[ tf.assign(w,v) for w,v in zip(current_vars, combined)])
            with tf.get_default_graph().control_dependencies([op5]):
                return tf.no_op()

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
  def variables(self):
      return self.optimizer.variables()
