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
import numpy as np
import inspect

class GradientMagnitudeOptimizer(optimizer.Optimizer):
  """Projects layer gradients to a norm then multiplies by a constant"""
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
    var_list = [v for _, v in grads_and_vars]
    grad_list = [g for g, _ in grads_and_vars]


    self._prepare()
    def project_gradient_layer(gs):
        if self.config.norm == 'softmax':
            return tf.nn.softmax(gs)
        elif self.config.norm == 'euclidean':
            return gs / (tf.sqrt(tf.reduce_sum(tf.square(gs)))+1e-8)
        elif self.config.norm == 'inf':
            return gs / (tf.norm(gs, ord=np.inf)+1e-8)
        elif self.config.norm == 'max':
            return gs / (tf.reduce_max(tf.abs(gs))+1e-8)
        elif self.config.norm == False:
            return gs
        else:
            return gs / (tf.norm(gs, ord=self.config.norm)+1e-8)

    lam = []
    for g, v in grads_and_vars:
        _lam = self.gan.configurable_param(self.config["lambda"])
        opts = self.gan.layer_options(v)
        if opts is not None:
            print("OPTS", opts)
            if "gradient_magnitude_lambda" in opts:
                _lam *= float(opts["gradient_magnitude_lambda"])
        lam.append(_lam)

    print("Lambdas = ", lam)

    def number_weights(v):
        count = np.prod(self.gan.ops.shape(v))
        return count
    if self.config.per_weight:
        newlam = []
        for _lam, v in zip(lam,grad_list):
            newlam.append(_lam * number_weights(v))
        lam = newlam
    newgrads = [_lam * project_gradient_layer(g) for _lam, g in zip(lam, grad_list)]
    newgrads_and_vars = list(zip(newgrads, var_list)).copy()
    op = self.optimizer.apply_gradients(newgrads_and_vars, global_step=global_step, name=name)
    with tf.get_default_graph().control_dependencies([op]):
        return tf.no_op()

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def variables(self):
      return super().variables() + self.optimizer.variables()
