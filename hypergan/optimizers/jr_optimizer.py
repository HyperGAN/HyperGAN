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

class JROptimizer(optimizer.Optimizer):
  """https://arxiv.org/pdf/1806.09235.pdf"""
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="SOSOptimizer", optimizer=None, alpha=1):
    super().__init__(use_locking, name)
    self._alpha = alpha
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate
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
    all_vars = [ v for _,v in grads_and_vars]
    d_vars = []
    g_vars = []
    all_grads = [ g for g, _ in grads_and_vars ]
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
        elif var in self.gan.g_vars():
            g_vars += [var]
        else:
            raise("Couldn't find var in g_vars or d_vars")

    with ops.init_scope():
        self.optimizer._create_slots([v for g,v in grads_and_vars])
    self._prepare()

    d_grads = all_grads[:len(d_vars)]
    g_grads = all_grads[len(d_vars):]
    dc_grads = [tf.square(tf.norm(d)) for d in d_grads]
    gc_grads = [tf.square(tf.norm(g)) for g in g_grads]
    gamma12 = tf.gradients(gc_grads, d_vars) + [tf.zeros_like(g) for g in g_vars]
    gamma21 = [tf.zeros_like(d) for d in d_vars] + tf.gradients(dc_grads, g_vars)
    gamma12 = [ tf.zeros_like(ddg) if _dg is None else _dg for ddg, _dg in zip(all_vars, gamma12) ]
    gamma21 = [ tf.zeros_like(ddg) if _dg is None else _dg for ddg, _dg in zip(all_vars, gamma21) ]
    __gamma12 = [ tf.reduce_sum(_gamma12) for _gamma12 in gamma12 ]
    __gamma21 = [ tf.reduce_sum(_gamma21) for _gamma21 in gamma21 ]
    #gamma12_metric = self.gan.ops.squash(sum(gamma12))
    gamma12_metric = self.gan.ops.squash(sum(__gamma12))
    self.gan.add_metric('gamma12', gamma12_metric)
    gamma21_metric = self.gan.ops.squash(sum(__gamma21))
    self.gan.add_metric('gamma21', gamma21_metric)
   
    new_grads = []
    for _gamma12, _gamma21, _grads in zip(gamma12, gamma21, all_grads):
        Eo = _grads - \
             0.5*self._alpha*_gamma21 +\
             0.5*self._alpha*_gamma12
        new_grads += [ Eo ]

    new_grads_and_vars = list(zip(new_grads, all_vars)).copy()

    return self.optimizer.apply_gradients(new_grads_and_vars, global_step=global_step, name=name)
  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
