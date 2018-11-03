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

class ConsensusOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="CurlOptimizer", optimizer=None, rho=1, beta=1, gamma=1):
    super().__init__(use_locking, name)
    self._beta = beta
    self._rho = rho
    self._gamma = gamma
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

  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    self.optimizer._create_slots(var_list)

  def _apply_dense(self, grad, var):
    return self.optimizer._apply_dense(grad, var)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    var_list = [ v for _,v in grads_and_vars]
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
    consensus_reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in all_grads[:len(d_vars)] if g is not None
    )
    Jgrads = tf.gradients(consensus_reg, d_vars) + [tf.zeros_like(g) for g in g_vars]
    print("LR_T", self._lr_t)
    op7 = tf.group([tf.assign_sub(v, self._lr_t*(grad+(jg * self._beta))) if jg is not None else tf.assign_sub(v,self._lr_t*grad+tf.zeros_like(v)) for v,grad, jg in zip(var_list, all_grads, Jgrads)])
    with tf.get_default_graph().control_dependencies([op7]):

        op6 = tf.group([self.optimizer._apply_dense(g,v) for g,v in grads_and_vars])
        with tf.get_default_graph().control_dependencies([op6]):
            return tf.no_op()

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
