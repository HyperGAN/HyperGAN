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

class NegativeMomentumOptimizer(optimizer.Optimizer):
  """Steps multiple times and decays results"""
  def __init__(self, learning_rate=0.001, decay=1.0, gan=None, config=None, use_locking=False, name="NegativeMomentumOptimizer", optimizer=None):
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

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    d_vars = []
    g_vars = []
    d_grads = []
    g_grads = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
            d_grads += [grad]
        elif var in self.gan.g_vars():
            g_vars += [var]
            g_grads += [grad]
        else:
            raise ValidationException("Couldn't find var in g_vars or d_vars " + var.name)
    grad_list = d_grads + g_grads
    var_list = d_vars + g_vars

    with ops.init_scope():
            nms = [self._get_or_make_slot(v, tf.zeros_like(v), "nm", self._name) for v in var_list]
    self._prepare()

    nms = [self.get_slot(v, "nm") for v in var_list]
    momentum = []
    for grad, nm, w in zip(grad_list, nms, var_list):
        momentum += [-self._decay * nm]

    newgrads = [g + m for g, m in zip(grad_list, momentum)]

    new_grads_and_vars = list(zip(newgrads, var_list)).copy()

    op2 = self.optimizer.apply_gradients(new_grads_and_vars, global_step=global_step, name=name)
    with tf.get_default_graph().control_dependencies([op2]):
        save = tf.group(*[tf.assign(nm, ((self.config.alpha or 0.666) *grad+ (1-self.config.beta or 0.5)*nm)) for nm, grad in zip(nms, grad_list)])
        with tf.get_default_graph().control_dependencies([save]):
            return tf.no_op()

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def _apply_dense(self, grad, var):
    raise NotImplementedError("_apply_dense not callable.")

  def variables(self):
      return self.optimizer.variables()
