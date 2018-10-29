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

class CurlOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="CurlOptimizer", optimizer=None):
    super().__init__(use_locking, name)
    self._p = p
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate
    def create_optimizer(klass, options):
        options['gan']=self.gan
        options['config']=options
        defn = {k: v for k, v in options.items() if k in inspect.getargspec(klass).args}
        return klass(options["learn_rate"], **defn)

    optimizer = hc.lookup_functions(optimizer)
    self.optimizer = create_optimizer(optimizer['class'], optimizer)
 
  def _prepare(self):
    super()._prepare()
    self.optimizer._prepare()
    if self._p == "rand":
        self._p_t = tf.random_uniform([1], minval=0.0, maxval=1.0)
    else:
        self._p_t = ops.convert_to_tensor(self._p, name="p")


  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    self.optimizer._create_slots(var_list)
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "oldv", self._name)
      
  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    p_t = math_ops.cast(self._p_t, var.dtype.base_dtype)

    v = self.get_slot(var, "oldv")
    reset_v = var.assign(v)
    store_v = v.assign(var)
    movement = grad * lr_t 
    var_update1 = state_ops.assign_sub(var, movement)

    if var in self.gan.d_vars():
        grad2 = tf.gradients(self.gan.trainer.d_loss, var)[0]
    elif var in self.gan.g_vars():
        grad2 = tf.gradients(self.gan.trainer.g_loss, var)[0]
    else:
        raise("Couldn't find var in g_vars or d_vars")

    grad3 = (grad - p_t * (grad2 - grad))

    var_update2 =self.optimizer._apply_dense(grad3, var)
    return control_flow_ops.group(*[store_v, var_update1, reset_v, var_update2])

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
