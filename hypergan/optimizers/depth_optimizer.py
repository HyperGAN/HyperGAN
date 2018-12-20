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

class DepthOptimizer(optimizer.Optimizer):
  """Steps multiple times and decays results"""
  def __init__(self, learning_rate=0.001, decay=0.9, gan=None, config=None, use_locking=False, name="depthOptimizer", optimizer=None, depth=3):
    super().__init__(use_locking, name)
    self._decay = decay
    self._depth = depth
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

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    d_vars = []
    g_vars = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
        elif var in self.gan.g_vars():
            g_vars += [var]
        else:
            raise Exception("Couldn't find var in g_vars or d_vars")

    if self.config.apply_on == "discriminator":
        depth_vars = d_vars
    else:
        depth_vars = d_vars + g_vars
    with ops.init_scope():
        [self._get_or_make_slot(v, v, "depth", self.name) for v in depth_vars]
        self.optimizer._create_slots([v for g,v in grads_and_vars])
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                self._zeros_slot(var, "depth", self.name)

    self._prepare()
    depth_slots = [self.get_slot(v, "depth") for v in depth_vars]
    for name in self.optimizer.get_slot_names():
        for var in self.optimizer.variables():
            depth_vars += [var]
            depth_slots += [self._zeros_slot(var, "depth", self.name)]

    def calculate_depth(grads_and_vars_k,k=0):
        if(k == 0):
            return tf.group(*[tf.assign(v,nv) for v,nv in zip(depth_vars, depth_slots)])

        op2 = self.optimizer.apply_gradients(grads_and_vars_k, global_step=global_step, name=name)
        with tf.get_default_graph().control_dependencies([op2]):
            w_k_combined = [self._decay *w_k_1 + (1.-self._decay)*w_hat for w_hat, w_k_1 in zip(depth_slots, depth_vars)]
            op3 = tf.group(*[tf.assign(w, v) for w,v in zip(depth_slots, w_k_combined)]) # store variables
            with tf.get_default_graph().control_dependencies([op3]):
                d_loss, g_loss = self.gan.loss.sample
                d_grads = tf.gradients(d_loss, d_vars)
                g_grads = tf.gradients(g_loss, g_vars)
                grads_k_1 = d_grads + g_grads
                grads_and_vars_k_1 = list(zip(grads_k_1,depth_vars)).copy()
                return calculate_depth(grads_and_vars_k_1,k-1)

    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(depth_slots, depth_vars)]) # store variables
    with tf.get_default_graph().control_dependencies([op1]):
        opd = calculate_depth(grads_and_vars, self._depth)
        with tf.get_default_graph().control_dependencies([opd]):
            return tf.no_op()

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def _apply_dense(self, grad, var):
    raise NotImplementedError("_apply_dense not callable.")

  def variables(self):
      return self.optimizer.variables()
