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

class OrthonormalOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="CurlOptimizer", optimizer=None, rho=1, beta=1, gamma=1):
    super().__init__(use_locking, name)
    self.gan = gan
    self.config = config

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
    flin = [ g for g,_ in grads_and_vars]
    d_vars = []
    g_vars = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
        elif var in self.gan.g_vars():
            g_vars += [var]
        else:
            raise("Couldn't find var in g_vars or d_vars")


    consensus_reg = [tf.square(g) for g in flin[:len(d_vars)] if g is not None]
    Jgrads = tf.gradients(consensus_reg, d_vars) + [tf.zeros_like(g) for g in g_vars]
    shapes = [self.gan.ops.shape(l) for l in flin]
    u = [tf.reshape(l, [-1]) for l in flin[:len(d_vars)]]
    v = [tf.reshape(l, [-1]) if l is not None else tf.reshape(tf.zeros_like(v), [-1]) for l,v in zip(Jgrads[:len(d_vars)], d_vars)]
    
    def proj(u, v,shape):
        dot = tf.tensordot(v, u, 1) / (tf.square(u)+1e-8)
        dot = tf.maximum(0.0, dot)
        dot = tf.minimum(1.0, dot)
        dot = dot * u
        dot = tf.reshape(dot, shape)
        return dot
    proj_u1_v2 = [proj(_u, _v, _s) for _u, _v, _s in zip(u, v, shapes)]
    flin = [_flin + self.gan.configurable_param(self.config.ortholambda) * proj for _flin, proj in zip(flin, proj_u1_v2)] + flin[len(d_vars):]

    step3 = list(zip(flin, var_list))
    op6 = self.optimizer.apply_gradients(step3.copy(), global_step=global_step, name=name)


    with tf.get_default_graph().control_dependencies([op6]):
        return tf.no_op()

                    # Flin = gamma * IF - rho * JF + beta * JtF
                    #op7 = tf.group(*[tf.assign_add(gsw, (jg * self._beta)) if jg is not None else tf.no_op() for gsw, jg in zip(gswap, Jgrads)])
                    #with tf.get_default_graph().control_dependencies([op7]):
                    #    flin_grads_and_vars = zip(gswap, var_list)
                    #    # step 1
                    #    op8 = self.optimizer.apply_gradients(list(flin_grads_and_vars).copy(), global_step=global_step, name=name)
                    #    with tf.get_default_graph().control_dependencies([op8]):
                    #        return tf.no_op()
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
  def variables(self):
      return super().variables() + self.optimizer.variables()
