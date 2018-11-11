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

class GradientMagnitudeOptimizer(optimizer.Optimizer):
  """Projects layer gradients to unit ball then multiplies by a constant"""
  def __init__(self, learning_rate=0.001, decay=0.9, gan=None, config=None, use_locking=False, name="EmaOptimizer", optimizer=None):
    super().__init__(use_locking, name)
    self._decay = decay
    self.gan = gan
    self.config = config
    self.name = name
    def create_optimizer(klass, options):
        options['gan']=self.gan
        options['config']=options
        defn = {k: v for k, v in options.items() if k in inspect.getargspec(klass).args}
        return klass(options.learn_rate, **defn)

    optimizer = hc.lookup_functions(optimizer)
    self.optimizer = create_optimizer(optimizer['class'], optimizer)
 
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

    lam = self.gan.configurable_param(self.config["lambda"])

    self._prepare()
    def project_gradient_layer(gs):
        return gs / (tf.sqrt(tf.reduce_sum(tf.square(gs)))+1e-8)

    newgrads = [lam * project_gradient_layer(g) for g, _ in grads_and_vars]
    newgrads_and_vars = list(zip(newgrads, var_list)).copy()
    op = self.optimizer.apply_gradients(newgrads_and_vars, global_step=global_step, name=name)
    with tf.get_default_graph().control_dependencies([op]):
        return tf.no_op()

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def variables(self):
      return super().variables() + self.optimizer.variables()
